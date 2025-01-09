# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/core/effects.py)
import typing

import numpy as np
import paddle

from . import util
from ._julius import SplitBands


class EffectMixin:
    GAIN_FACTOR = np.log(10) / 20
    """Gain factor for converting between amplitude and decibels."""
    CODEC_PRESETS = {
        "8-bit": {
            "format": "wav",
            "encoding": "ULAW",
            "bits_per_sample": 8
        },
        "GSM-FR": {
            "format": "gsm"
        },
        "MP3": {
            "format": "mp3",
            "compression": -9
        },
        "Vorbis": {
            "format": "vorbis",
            "compression": -1
        },
        "Ogg": {
            "format": "ogg",
            "compression": -1,
        },
        "Amr-nb": {
            "format": "amr-nb"
        },
    }
    """Presets for applying codecs via torchaudio."""

    def mix(
            self,
            other,
            snr: typing.Union[paddle.Tensor, np.ndarray, float]=10,
            other_eq: typing.Union[paddle.Tensor, np.ndarray]=None, ):
        """Mixes noise with signal at specified
        signal-to-noise ratio. Optionally, the
        other signal can be equalized in-place.


        Parameters
        ----------
        other : AudioSignal
            AudioSignal object to mix with.
        snr : typing.Union[paddle.Tensor, np.ndarray, float], optional
            Signal to noise ratio, by default 10
        other_eq : typing.Union[paddle.Tensor, np.ndarray], optional
            EQ curve to apply to other signal, if any, by default None

        Returns
        -------
        AudioSignal
            In-place modification of AudioSignal.
        """
        snr = util.ensure_tensor(snr)

        pad_len = max(0, self.signal_length - other.signal_length)
        other.zero_pad(0, pad_len)
        other.truncate_samples(self.signal_length)
        if other_eq is not None:
            other = other.equalizer(other_eq)

        tgt_loudness = self.loudness() - snr
        other = other.normalize(tgt_loudness)

        self.audio_data = self.audio_data + other.audio_data
        return self

    def convolve(self, other, start_at_max: bool=True):
        """Convolves self with other.
        This function uses FFTs to do the convolution.

        Parameters
        ----------
        other : AudioSignal
            Signal to convolve with.
        start_at_max : bool, optional
            Whether to start at the max value of other signal, to
            avoid inducing delays, by default True

        Returns
        -------
        AudioSignal
            Convolved signal, in-place.
        """
        from . import AudioSignal

        pad_len = self.signal_length - other.signal_length

        if pad_len > 0:
            other.zero_pad(0, pad_len)
        else:
            other.truncate_samples(self.signal_length)

        if start_at_max:
            # Use roll to rotate over the max for every item
            # so that the impulse responses don't induce any
            # delay.
            idx = paddle.argmax(paddle.abs(other.audio_data), axis=-1)
            irs = paddle.zeros_like(other.audio_data)
            for i in range(other.batch_size):
                irs[i] = paddle.roll(
                    other.audio_data[i], shifts=-idx[i].item(), axis=-1)
            other = AudioSignal(irs, other.sample_rate)

        delta = paddle.zeros_like(other.audio_data)
        delta[..., 0] = 1

        length = self.signal_length
        delta_fft = paddle.fft.rfft(delta, n=length)
        other_fft = paddle.fft.rfft(other.audio_data, n=length)
        self_fft = paddle.fft.rfft(self.audio_data, n=length)

        convolved_fft = other_fft * self_fft
        convolved_audio = paddle.fft.irfft(convolved_fft, n=length)

        delta_convolved_fft = other_fft * delta_fft
        delta_audio = paddle.fft.irfft(delta_convolved_fft, n=length)

        # Use the delta to rescale the audio exactly as needed.
        delta_max = paddle.max(paddle.abs(delta_audio), axis=-1, keepdim=True)
        scale = 1 / paddle.clip(delta_max, min=1e-5)
        convolved_audio = convolved_audio * scale

        self.audio_data = convolved_audio

        return self

    def apply_ir(
            self,
            ir,
            drr: typing.Union[paddle.Tensor, np.ndarray, float]=None,
            ir_eq: typing.Union[paddle.Tensor, np.ndarray]=None,
            use_original_phase: bool=False, ):
        """Applies an impulse response to the signal. If ` is`ir_eq``
        is specified, the impulse response is equalized before
        it is applied, using the given curve.

        Parameters
        ----------
        ir : AudioSignal
            Impulse response to convolve with.
        drr : typing.Union[paddle.Tensor, np.ndarray, float], optional
            Direct-to-reverberant ratio that impulse response will be
            altered to, if specified, by default None
        ir_eq : typing.Union[paddle.Tensor, np.ndarray], optional
            Equalization that will be applied to impulse response
            if specified, by default None
        use_original_phase : bool, optional
            Whether to use the original phase, instead of the convolved
            phase, by default False

        Returns
        -------
        AudioSignal
            Signal with impulse response applied to it
        """
        if ir_eq is not None:
            ir = ir.equalizer(ir_eq)
        if drr is not None:
            ir = ir.alter_drr(drr)

        # Save the peak before
        max_spk = self.audio_data.abs().max(axis=-1, keepdim=True)

        # Augment the impulse response to simulate microphone effects
        # and with varying direct-to-reverberant ratio.
        phase = self.phase
        self.convolve(ir)

        # Use the input phase
        if use_original_phase:
            self.stft()
            self.stft_data = self.magnitude * util.exp_compat(1j * phase)
            self.istft()

        # Rescale to the input's amplitude
        max_transformed = self.audio_data.abs().max(axis=-1, keepdim=True)
        scale_factor = max_spk.clip(1e-8) / max_transformed.clip(1e-8)
        self = self * scale_factor

        return self

    def ensure_max_of_audio(self, _max: float=1.0):
        """Ensures that ``abs(audio_data) <= max``.

        Parameters
        ----------
        max : float, optional
            Max absolute value of signal, by default 1.0

        Returns
        -------
        AudioSignal
            Signal with values scaled between -max and max.
        """
        peak = self.audio_data.abs().max(axis=-1, keepdim=True)
        peak_gain = paddle.ones_like(peak)
        # peak_gain[peak > _max] = _max / peak[peak > _max]
        peak_gain = paddle.where(peak > _max, _max / peak, peak_gain)
        self.audio_data = self.audio_data * peak_gain
        return self

    def normalize(self,
                  db: typing.Union[paddle.Tensor, np.ndarray, float]=-24.0):
        """Normalizes the signal's volume to the specified db, in LUFS.
        This is GPU-compatible, making for very fast loudness normalization.

        Parameters
        ----------
        db : typing.Union[paddle.Tensor, np.ndarray, float], optional
            Loudness to normalize to, by default -24.0

        Returns
        -------
        AudioSignal
            Normalized audio signal.
        """
        db = util.ensure_tensor(db)
        ref_db = self.loudness()
        gain = db.astype(ref_db.dtype) - ref_db
        gain = util.exp_compat(gain * self.GAIN_FACTOR)

        self.audio_data = self.audio_data * gain[:, None, None]
        return self

    def volume_change(self, db: typing.Union[paddle.Tensor, np.ndarray, float]):
        """Change volume of signal by some amount, in dB.

        Parameters
        ----------
        db : typing.Union[paddle.Tensor, np.ndarray, float]
            Amount to change volume by.

        Returns
        -------
        AudioSignal
            Signal at new volume.
        """
        db = util.ensure_tensor(db, ndim=1)
        gain = util.exp_compat(db * self.GAIN_FACTOR)
        self.audio_data = self.audio_data * gain[:, None, None]
        return self

    def mel_filterbank(self, n_bands: int):
        """Breaks signal into mel bands.

        Parameters
        ----------
        n_bands : int
            Number of mel bands to use.

        Returns
        -------
        paddle.Tensor
            Mel-filtered bands, with last axis being the band index.
        """
        filterbank = SplitBands(self.sample_rate, n_bands)
        filtered = filterbank(self.audio_data)
        return filtered.transpose([1, 2, 3, 0])

    def equalizer(self, db: typing.Union[paddle.Tensor, np.ndarray]):
        """Applies a mel-spaced equalizer to the audio signal.

        Parameters
        ----------
        db : typing.Union[paddle.Tensor, np.ndarray]
            EQ curve to apply.

        Returns
        -------
        AudioSignal
            AudioSignal with equalization applied.
        """
        db = util.ensure_tensor(db)
        n_bands = db.shape[-1]
        fbank = self.mel_filterbank(n_bands)

        # If there's a batch dimension, make sure it's the same.
        if db.ndim == 2:
            if db.shape[0] != 1:
                assert db.shape[0] == fbank.shape[0]
        else:
            db = db.unsqueeze(0)

        weights = (10**db).astype("float32")
        fbank = fbank * weights[:, None, None, :]
        eq_audio_data = fbank.sum(-1)
        self.audio_data = eq_audio_data
        return self

    def clip_distortion(
            self,
            clip_percentile: typing.Union[paddle.Tensor, np.ndarray, float]):
        """Clips the signal at a given percentile. The higher it is,
        the lower the threshold for clipping.

        Parameters
        ----------
        clip_percentile : typing.Union[paddle.Tensor, np.ndarray, float]
            Values are between 0.0 to 1.0. Typical values are 0.1 or below.

        Returns
        -------
        AudioSignal
            Audio signal with clipped audio data.
        """
        clip_percentile = util.ensure_tensor(clip_percentile, ndim=1)
        clip_percentile = clip_percentile.cpu().numpy()
        min_thresh = paddle.quantile(
            self.audio_data, (clip_percentile / 2).tolist(), axis=-1)[None]
        max_thresh = paddle.quantile(
            self.audio_data, (1 - clip_percentile / 2).tolist(), axis=-1)[None]

        nc = self.audio_data.shape[1]
        min_thresh = min_thresh[:, :nc, :]
        max_thresh = max_thresh[:, :nc, :]

        self.audio_data = self.audio_data.clip(min_thresh, max_thresh)

        return self

    def quantization(self,
                     quantization_channels: typing.Union[paddle.Tensor,
                                                         np.ndarray, int]):
        """Applies quantization to the input waveform.

        Parameters
        ----------
        quantization_channels : typing.Union[paddle.Tensor, np.ndarray, int]
            Number of evenly spaced quantization channels to quantize
            to.

        Returns
        -------
        AudioSignal
            Quantized AudioSignal.
        """
        quantization_channels = util.ensure_tensor(
            quantization_channels, ndim=3)

        x = self.audio_data
        quantization_channels = quantization_channels.astype(x.dtype)
        x = (x + 1) / 2
        x = x * quantization_channels
        x = x.floor()
        x = x / quantization_channels
        x = 2 * x - 1

        residual = (self.audio_data - x).detach()
        self.audio_data = self.audio_data - residual
        return self

    def mulaw_quantization(self,
                           quantization_channels: typing.Union[
                               paddle.Tensor, np.ndarray, int]):
        """Applies mu-law quantization to the input waveform.

        Parameters
        ----------
        quantization_channels : typing.Union[paddle.Tensor, np.ndarray, int]
            Number of mu-law spaced quantization channels to quantize
            to.

        Returns
        -------
        AudioSignal
            Quantized AudioSignal.
        """
        mu = quantization_channels - 1.0
        mu = util.ensure_tensor(mu, ndim=3)

        x = self.audio_data

        # quantize
        x = paddle.sign(x) * paddle.log1p(mu * paddle.abs(x)) / paddle.log1p(mu)
        x = ((x + 1) / 2 * mu + 0.5).astype("int64")

        # unquantize
        x = (x.astype(mu.dtype) / mu) * 2 - 1.0
        x = paddle.sign(x) * (
            util.exp_compat(paddle.abs(x) * paddle.log1p(mu)) - 1.0) / mu

        residual = (self.audio_data - x).detach()
        self.audio_data = self.audio_data - residual
        return self

    def __matmul__(self, other):
        return self.convolve(other)


class ImpulseResponseMixin:
    """These functions are generally only used with AudioSignals that are derived
    from impulse responses, not other sources like music or speech. These methods
    are used to replicate the data augmentation described in [1].

    1.  Bryan, Nicholas J. "Impulse response data augmentation and deep
        neural networks for blind room acoustic parameter estimation."
        ICASSP 2020-2020 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP). IEEE, 2020.
    """

    def decompose_ir(self):
        """Decomposes an impulse response into early and late
        field responses.
        """
        # Equations 1 and 2
        # -----------------
        # Breaking up into early
        # response + late field response.

        td = paddle.argmax(self.audio_data, axis=-1, keepdim=True)
        t0 = int(self.sample_rate * 0.0025)

        idx = paddle.arange(self.audio_data.shape[-1])[None, None, :]
        idx = idx.expand([self.batch_size, -1, -1])
        early_idx = (idx >= td - t0) * (idx <= td + t0)

        early_response = paddle.zeros_like(self.audio_data)

        # early_response[early_idx] = self.audio_data[early_idx]
        early_response = paddle.where(early_idx, self.audio_data,
                                      early_response)

        late_idx = ~early_idx
        late_field = paddle.zeros_like(self.audio_data)
        # late_field[late_idx] = self.audio_data[late_idx]
        late_field = paddle.where(late_idx, self.audio_data, late_field)

        # Equation 4
        # ----------
        # Decompose early response into windowed
        # direct path and windowed residual.

        window = paddle.zeros_like(self.audio_data)
        window_idx = paddle.nonzero(early_idx)
        for idx in range(self.batch_size):
            # window_idx = early_idx[idx, 0]

            # ----- Just for this -----
            # window[idx, ..., window_idx] = self.get_window("hann", window_idx.sum().item())
            # indices = paddle.nonzero(window_idx).reshape(
            #     [-1])  # shape: [num_true], dtype: int64  
            indices = window_idx[window_idx[:, 0] == idx][:, -1]

            temp_window = self.get_window("hann", indices.shape[0])

            window_slice = window[idx, 0]
            updated_window_slice = paddle.scatter(
                window_slice, index=indices, updates=temp_window)

            window[idx, 0] = updated_window_slice
            # ----- Just for that -----

        return early_response, late_field, window

    def measure_drr(self):
        """Measures the direct-to-reverberant ratio of the impulse
        response.

        Returns
        -------
        float
            Direct-to-reverberant ratio
        """
        early_response, late_field, _ = self.decompose_ir()
        num = (early_response**2).sum(axis=-1)
        den = (late_field**2).sum(axis=-1)
        drr = 10 * paddle.log10(num / den)
        return drr

    @staticmethod
    def solve_alpha(early_response, late_field, wd, target_drr):
        """Used to solve for the alpha value, which is used
        to alter the drr.
        """
        # Equation 5
        # ----------
        # Apply the good ol' quadratic formula.

        wd_sq = wd**2
        wd_sq_1 = (1 - wd)**2
        e_sq = early_response**2
        l_sq = late_field**2
        a = (wd_sq * e_sq).sum(axis=-1)
        b = (2 * (1 - wd) * wd * e_sq).sum(axis=-1)
        c = (wd_sq_1 * e_sq).sum(axis=-1) - paddle.pow(10 * paddle.ones_like(
            target_drr, dtype="float32"), target_drr.cast("float32") /
                                                       10) * l_sq.sum(axis=-1)

        expr = ((b**2) - 4 * a * c).sqrt()
        alpha = paddle.maximum(
            (-b - expr) / (2 * a),
            (-b + expr) / (2 * a), )
        return alpha

    def alter_drr(self, drr: typing.Union[paddle.Tensor, np.ndarray, float]):
        """Alters the direct-to-reverberant ratio of the impulse response.

        Parameters
        ----------
        drr : typing.Union[paddle.Tensor, np.ndarray, float]
            Direct-to-reverberant ratio that impulse response will be
            altered to, if specified, by default None

        Returns
        -------
        AudioSignal
            Altered impulse response.
        """
        drr = util.ensure_tensor(
            drr, 2, self.batch_size
        )  # Assuming util.ensure_tensor is adapted or equivalent exists

        early_response, late_field, window = self.decompose_ir()
        alpha = self.solve_alpha(early_response, late_field, window, drr)
        min_alpha = late_field.abs().max(axis=-1)[0] / early_response.abs().max(
            axis=-1)[0]
        alpha = paddle.maximum(alpha, min_alpha)[..., None]

        aug_ir_data = alpha * window * early_response + (
            (1 - window) * early_response) + late_field
        self.audio_data = aug_ir_data
        self.ensure_max_of_audio(
        )  # Assuming ensure_max_of_audio is a method defined elsewhere
        return self
