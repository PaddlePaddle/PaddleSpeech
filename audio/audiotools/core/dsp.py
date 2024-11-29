import typing

import numpy as np
import paddle

from . import _julius
from . import util


class DSPMixin:
    _original_batch_size = None
    _original_num_channels = None
    _padded_signal_length = None

    # def _preprocess_signal_for_windowing(self, window_duration, hop_duration):
    #     self._original_batch_size = self.batch_size
    #     self._original_num_channels = self.num_channels

    #     window_length = int(window_duration * self.sample_rate)
    #     hop_length = int(hop_duration * self.sample_rate)

    #     if window_length % hop_length != 0:
    #         factor = window_length // hop_length
    #         window_length = factor * hop_length

    #     self.zero_pad(hop_length, hop_length)
    #     self._padded_signal_length = self.signal_length

    #     return window_length, hop_length

    # def windows(
    #     self, window_duration: float, hop_duration: float, preprocess: bool = True
    # ):
    #     """Generator which yields windows of specified duration from signal with a specified
    #     hop length.

    #     Parameters
    #     ----------
    #     window_duration : float
    #         Duration of every window in seconds.
    #     hop_duration : float
    #         Hop between windows in seconds.
    #     preprocess : bool, optional
    #         Whether to preprocess the signal, so that the first sample is in
    #         the middle of the first window, by default True

    #     Yields
    #     ------
    #     AudioSignal
    #         Each window is returned as an AudioSignal.
    #     """
    #     if preprocess:
    #         window_length, hop_length = self._preprocess_signal_for_windowing(
    #             window_duration, hop_duration
    #         )

    #     self.audio_data = self.audio_data.reshape(-1, 1, self.signal_length)

    #     for b in range(self.batch_size):
    #         i = 0
    #         start_idx = i * hop_length
    #         while True:
    #             start_idx = i * hop_length
    #             i += 1
    #             end_idx = start_idx + window_length
    #             if end_idx > self.signal_length:
    #                 break
    #             yield self[b, ..., start_idx:end_idx]

    # def collect_windows(
    #     self, window_duration: float, hop_duration: float, preprocess: bool = True
    # ):
    #     """Reshapes signal into windows of specified duration from signal with a specified
    #     hop length. Window are placed along the batch dimension. Use with
    #     :py:func:`audiotools.core.dsp.DSPMixin.overlap_and_add` to reconstruct the
    #     original signal.

    #     Parameters
    #     ----------
    #     window_duration : float
    #         Duration of every window in seconds.
    #     hop_duration : float
    #         Hop between windows in seconds.
    #     preprocess : bool, optional
    #         Whether to preprocess the signal, so that the first sample is in
    #         the middle of the first window, by default True

    #     Returns
    #     -------
    #     AudioSignal
    #         AudioSignal unfolded with shape ``(nb * nch * num_windows, 1, window_length)``
    #     """
    #     if preprocess:
    #         window_length, hop_length = self._preprocess_signal_for_windowing(
    #             window_duration, hop_duration
    #         )

    #     # self.audio_data: (nb, nch, nt).
    #     unfolded = paddle.nn.functional.unfold(
    #         self.audio_data.reshape(-1, 1, 1, self.signal_length),
    #         kernel_size=(1, window_length),
    #         stride=(1, hop_length),
    #     )
    #     # unfolded: (nb * nch, window_length, num_windows).
    #     # -> (nb * nch * num_windows, 1, window_length)
    #     unfolded = unfolded.permute(0, 2, 1).reshape(-1, 1, window_length)
    #     self.audio_data = unfolded
    #     return self

    # def overlap_and_add(self, hop_duration: float):
    #     """Function which takes a list of windows and overlap adds them into a
    #     signal the same length as ``audio_signal``.

    #     Parameters
    #     ----------
    #     hop_duration : float
    #         How much to shift for each window
    #         (overlap is window_duration - hop_duration) in seconds.

    #     Returns
    #     -------
    #     AudioSignal
    #         overlap-and-added signal.
    #     """
    #     hop_length = int(hop_duration * self.sample_rate)
    #     window_length = self.signal_length

    #     nb, nch = self._original_batch_size, self._original_num_channels

    #     unfolded = self.audio_data.reshape(nb * nch, -1, window_length).permute(0, 2, 1)
    #     folded = paddle.nn.functional.fold(
    #         unfolded,
    #         output_size=(1, self._padded_signal_length),
    #         kernel_size=(1, window_length),
    #         stride=(1, hop_length),
    #     )

    #     norm = paddle.ones_like(unfolded, device=unfolded.device)
    #     norm = paddle.nn.functional.fold(
    #         norm,
    #         output_size=(1, self._padded_signal_length),
    #         kernel_size=(1, window_length),
    #         stride=(1, hop_length),
    #     )

    #     folded = folded / norm

    #     folded = folded.reshape(nb, nch, -1)
    #     self.audio_data = folded
    #     self.trim(hop_length, hop_length)
    #     return self

    def low_pass(self,
                 cutoffs: typing.Union[paddle.Tensor, np.ndarray, float],
                 zeros: int=51):
        """Low-passes the signal in-place. Each item in the batch
        can have a different low-pass cutoff, if the input
        to this signal is an array or tensor. If a float, all
        items are given the same low-pass filter.

        Parameters
        ----------
        cutoffs : typing.Union[paddle.Tensor, np.ndarray, float]
            Cutoff in Hz of low-pass filter.
        zeros : int, optional
            Number of taps to use in low-pass filter, by default 51

        Returns
        -------
        AudioSignal
            Low-passed AudioSignal.
        """
        cutoffs = util.ensure_tensor(cutoffs, 2, self.batch_size)
        cutoffs = cutoffs / self.sample_rate
        filtered = paddle.empty_like(self.audio_data)

        for i, cutoff in enumerate(cutoffs):
            lp_filter = _julius.LowPassFilter(cutoff.cpu(), zeros=zeros)
            filtered[i] = lp_filter(self.audio_data[i])

        self.audio_data = filtered
        self.stft_data = None
        return self

    def high_pass(self,
                  cutoffs: typing.Union[paddle.Tensor, np.ndarray, float],
                  zeros: int=51):
        """High-passes the signal in-place. Each item in the batch
        can have a different high-pass cutoff, if the input
        to this signal is an array or tensor. If a float, all
        items are given the same high-pass filter.

        Parameters
        ----------
        cutoffs : typing.Union[paddle.Tensor, np.ndarray, float]
            Cutoff in Hz of high-pass filter.
        zeros : int, optional
            Number of taps to use in high-pass filter, by default 51

        Returns
        -------
        AudioSignal
            High-passed AudioSignal.
        """
        cutoffs = util.ensure_tensor(cutoffs, 2, self.batch_size)
        cutoffs = cutoffs / self.sample_rate
        filtered = paddle.empty_like(self.audio_data)

        for i, cutoff in enumerate(cutoffs):
            hp_filter = _julius.HighPassFilter(cutoff.cpu(), zeros=zeros)
            filtered[i] = hp_filter(self.audio_data[i])

        self.audio_data = filtered
        self.stft_data = None
        return self

    # def mask_frequencies(
    #     self,
    #     fmin_hz: typing.Union[paddle.Tensor, np.ndarray, float],
    #     fmax_hz: typing.Union[paddle.Tensor, np.ndarray, float],
    #     val: float = 0.0,
    # ):
    #     """Masks frequencies between ``fmin_hz`` and ``fmax_hz``, and fills them
    #     with the value specified by ``val``. Useful for implementing SpecAug.
    #     The min and max can be different for every item in the batch.

    #     Parameters
    #     ----------
    #     fmin_hz : typing.Union[paddle.Tensor, np.ndarray, float]
    #         Lower end of band to mask out.
    #     fmax_hz : typing.Union[paddle.Tensor, np.ndarray, float]
    #         Upper end of band to mask out.
    #     val : float, optional
    #         Value to fill in, by default 0.0

    #     Returns
    #     -------
    #     AudioSignal
    #         Signal with ``stft_data`` manipulated. Apply ``.istft()`` to get the
    #         masked audio data.
    #     """
    #     # SpecAug
    #     mag, phase = self.magnitude, self.phase
    #     fmin_hz = util.ensure_tensor(fmin_hz, ndim=mag.ndim)
    #     fmax_hz = util.ensure_tensor(fmax_hz, ndim=mag.ndim)
    #     assert paddle.all(fmin_hz < fmax_hz)

    #     # build mask
    #     nbins = mag.shape[-2]
    #     bins_hz = paddle.linspace(0, self.sample_rate / 2, nbins, device=self.device)
    #     bins_hz = bins_hz[None, None, :, None].repeat(
    #         self.batch_size, 1, 1, mag.shape[-1]
    #     )
    #     mask = (fmin_hz <= bins_hz) & (bins_hz < fmax_hz)
    #     mask = mask.to(self.device)

    #     mag = mag.masked_fill(mask, val)
    #     phase = phase.masked_fill(mask, val)
    #     self.stft_data = mag * paddle.exp(1j * phase)
    #     return self

    # def mask_timesteps(
    #     self,
    #     tmin_s: typing.Union[paddle.Tensor, np.ndarray, float],
    #     tmax_s: typing.Union[paddle.Tensor, np.ndarray, float],
    #     val: float = 0.0,
    # ):
    #     """Masks timesteps between ``tmin_s`` and ``tmax_s``, and fills them
    #     with the value specified by ``val``. Useful for implementing SpecAug.
    #     The min and max can be different for every item in the batch.

    #     Parameters
    #     ----------
    #     tmin_s : typing.Union[paddle.Tensor, np.ndarray, float]
    #         Lower end of timesteps to mask out.
    #     tmax_s : typing.Union[paddle.Tensor, np.ndarray, float]
    #         Upper end of timesteps to mask out.
    #     val : float, optional
    #         Value to fill in, by default 0.0

    #     Returns
    #     -------
    #     AudioSignal
    #         Signal with ``stft_data`` manipulated. Apply ``.istft()`` to get the
    #         masked audio data.
    #     """
    #     # SpecAug
    #     mag, phase = self.magnitude, self.phase
    #     tmin_s = util.ensure_tensor(tmin_s, ndim=mag.ndim)
    #     tmax_s = util.ensure_tensor(tmax_s, ndim=mag.ndim)

    #     assert paddle.all(tmin_s < tmax_s)

    #     # build mask
    #     nt = mag.shape[-1]
    #     bins_t = paddle.linspace(0, self.signal_duration, nt, device=self.device)
    #     bins_t = bins_t[None, None, None, :].repeat(
    #         self.batch_size, 1, mag.shape[-2], 1
    #     )
    #     mask = (tmin_s <= bins_t) & (bins_t < tmax_s)

    #     mag = mag.masked_fill(mask, val)
    #     phase = phase.masked_fill(mask, val)
    #     self.stft_data = mag * paddle.exp(1j * phase)
    #     return self

    # def mask_low_magnitudes(
    #     self, db_cutoff: typing.Union[paddle.Tensor, np.ndarray, float], val: float = 0.0
    # ):
    #     """Mask away magnitudes below a specified threshold, which
    #     can be different for every item in the batch.

    #     Parameters
    #     ----------
    #     db_cutoff : typing.Union[paddle.Tensor, np.ndarray, float]
    #         Decibel value for which things below it will be masked away.
    #     val : float, optional
    #         Value to fill in for masked portions, by default 0.0

    #     Returns
    #     -------
    #     AudioSignal
    #         Signal with ``stft_data`` manipulated. Apply ``.istft()`` to get the
    #         masked audio data.
    #     """
    #     mag = self.magnitude
    #     log_mag = self.log_magnitude()

    #     db_cutoff = util.ensure_tensor(db_cutoff, ndim=mag.ndim)
    #     mask = log_mag < db_cutoff
    #     mag = mag.masked_fill(mask, val)

    #     self.magnitude = mag
    #     return self

    # def shift_phase(self, shift: typing.Union[paddle.Tensor, np.ndarray, float]):
    #     """Shifts the phase by a constant value.

    #     Parameters
    #     ----------
    #     shift : typing.Union[paddle.Tensor, np.ndarray, float]
    #         What to shift the phase by.

    #     Returns
    #     -------
    #     AudioSignal
    #         Signal with ``stft_data`` manipulated. Apply ``.istft()`` to get the
    #         masked audio data.
    #     """
    #     shift = util.ensure_tensor(shift, ndim=self.phase.ndim)
    #     self.phase = self.phase + shift
    #     return self

    # def corrupt_phase(self, scale: typing.Union[paddle.Tensor, np.ndarray, float]):
    #     """Corrupts the phase randomly by some scaled value.

    #     Parameters
    #     ----------
    #     scale : typing.Union[paddle.Tensor, np.ndarray, float]
    #         Standard deviation of noise to add to the phase.

    #     Returns
    #     -------
    #     AudioSignal
    #         Signal with ``stft_data`` manipulated. Apply ``.istft()`` to get the
    #         masked audio data.
    #     """
    #     scale = util.ensure_tensor(scale, ndim=self.phase.ndim)
    #     self.phase = self.phase + scale * paddle.randn_like(self.phase)
    #     return self

    # def preemphasis(self, coef: float = 0.85):
    #     """Applies pre-emphasis to audio signal.

    #     Parameters
    #     ----------
    #     coef : float, optional
    #         How much pre-emphasis to apply, lower values do less. 0 does nothing.
    #         by default 0.85

    #     Returns
    #     -------
    #     AudioSignal
    #         Pre-emphasized signal.
    #     """
    #     kernel = paddle.to_tensor([1, -coef, 0]).view(1, 1, -1).to(self.device)
    #     x = self.audio_data.reshape(-1, 1, self.signal_length)
    #     x = paddle.nn.functional.conv1d(x, kernel, padding=1)
    #     self.audio_data = x.reshape(*self.audio_data.shape)
    #     return self
