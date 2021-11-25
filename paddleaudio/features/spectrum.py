# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from functools import partial
from typing import Optional
from typing import Union

import paddle
import paddle.nn as nn

from .window import get_window

__all__ = [
    'Spectrogram',
    'MelSpectrogram',
    'LogMelSpectrogram',
]


def hz_to_mel(freq: Union[paddle.Tensor, float],
              htk: bool=False) -> Union[paddle.Tensor, float]:
    """Convert Hz to Mels.
    Parameters:
        freq: the input tensor of arbitrary shape, or a single floating point number.
        htk: use HTK formula to do the conversion.
            The default value is False.
    Returns:
        The frequencies represented in Mel-scale.
    """

    if htk:
        if isinstance(freq, paddle.Tensor):
            return 2595.0 * paddle.log10(1.0 + freq / 700.0)
        else:
            return 2595.0 * math.log10(1.0 + freq / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if isinstance(freq, paddle.Tensor):
        target = min_log_mel + paddle.log(
            freq / min_log_hz + 1e-10) / logstep  # prevent nan with 1e-10
        mask = (freq > min_log_hz).astype(freq.dtype)
        mels = target * mask + mels * (
            1 - mask)  # will replace by masked_fill OP in future
    else:
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz + 1e-10) / logstep

    return mels


def mel_to_hz(mel: Union[float, paddle.Tensor],
              htk: bool=False) -> Union[float, paddle.Tensor]:
    """Convert mel bin numbers to frequencies.
    Parameters:
        mel: the mel frequency represented as a tensor of arbitrary shape, or a floating point number.
        htk: use HTK formula to do the conversion.
    Returns:
        The frequencies represented in hz.
    """
    if htk:
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mel
    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region
    if isinstance(mel, paddle.Tensor):
        target = min_log_hz * paddle.exp(logstep * (mel - min_log_mel))
        mask = (mel > min_log_mel).astype(mel.dtype)
        freqs = target * mask + freqs * (
            1 - mask)  # will replace by masked_fill OP in future
    else:
        if mel >= min_log_mel:
            freqs = min_log_hz * math.exp(logstep * (mel - min_log_mel))

    return freqs


def mel_frequencies(n_mels: int=64,
                    f_min: float=0.0,
                    f_max: float=11025.0,
                    htk: bool=False,
                    dtype: str=paddle.float32):
    """Compute mel frequencies.
    Parameters:
        n_mels(int): number of Mel bins.
        f_min(float): the lower cut-off frequency, below which the filter response is zero.
        f_max(float): the upper cut-off frequency, above which the filter response is zero.
        htk(bool): whether to use htk formula.
        dtype(str): the datatype of the return frequencies.
    Returns:
        The frequencies represented in Mel-scale
    """
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(f_min, htk=htk)
    max_mel = hz_to_mel(f_max, htk=htk)
    mels = paddle.linspace(min_mel, max_mel, n_mels, dtype=dtype)
    freqs = mel_to_hz(mels, htk=htk)
    return freqs


def fft_frequencies(sr: int, n_fft: int, dtype: str=paddle.float32):
    """Compute fourier frequencies.
    Parameters:
        sr(int): the audio sample rate.
        n_fft(float): the number of fft bins.
        dtype(str): the datatype of the return frequencies.
    Returns:
        The frequencies represented in hz.
    """
    return paddle.linspace(0, float(sr) / 2, int(1 + n_fft // 2), dtype=dtype)


def compute_fbank_matrix(sr: int,
                         n_fft: int,
                         n_mels: int=64,
                         f_min: float=0.0,
                         f_max: Optional[float]=None,
                         htk: bool=False,
                         norm: Union[str, float]='slaney',
                         dtype: str=paddle.float32):
    """Compute fbank matrix.
    Parameters:
        sr(int): the audio sample rate.
        n_fft(int): the number of fft bins.
        n_mels(int): the number of Mel bins.
        f_min(float): the lower cut-off frequency, below which the filter response is zero.
        f_max(float): the upper cut-off frequency, above which the filter response is zero.
        htk: whether to use htk formula.
        return_complex(bool): whether to return complex matrix. If True, the matrix will
            be complex type. Otherwise, the real and image part will be stored in the last
            axis of returned tensor.
        dtype(str): the datatype of the returned fbank matrix.
    Returns:
        The fbank matrix of shape (n_mels, int(1+n_fft//2)).
    Shape:
        output: (n_mels, int(1+n_fft//2))
    """

    if f_max is None:
        f_max = float(sr) / 2

    # Initialize the weights
    weights = paddle.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft, dtype=dtype)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(
        n_mels + 2, f_min=f_min, f_max=f_max, htk=htk, dtype=dtype)

    fdiff = mel_f[1:] - mel_f[:-1]  #np.diff(mel_f)
    ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)
    #ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = paddle.maximum(
            paddle.zeros_like(lower), paddle.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    if norm == 'slaney':
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm.unsqueeze(1)
    elif isinstance(norm, int) or isinstance(norm, float):
        weights = paddle.nn.functional.normalize(weights, p=norm, axis=-1)

    return weights


def power_to_db(magnitude: paddle.Tensor,
                ref_value: float=1.0,
                amin: float=1e-10,
                top_db: Optional[float]=None) -> paddle.Tensor:
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units.
    The function computes the scaling ``10 * log10(x / ref)`` in a numerically
    stable way.
    Parameters:
        magnitude(Tensor): the input magnitude tensor of any shape.
        ref_value(float): the reference value. If smaller than 1.0, the db level
            of the signal will be pulled up accordingly. Otherwise, the db level
            is pushed down.
        amin(float): the minimum value of input magnitude, below which the input
            magnitude is clipped(to amin).
        top_db(float): the maximum db value of resulting spectrum, above which the
            spectrum is clipped(to top_db).
    Returns:
        The spectrogram in log-scale.
    shape:
        input: any shape
        output: same as input
    """
    if amin <= 0:
        raise Exception("amin must be strictly positive")

    if ref_value <= 0:
        raise Exception("ref_value must be strictly positive")

    ones = paddle.ones_like(magnitude)
    log_spec = 10.0 * paddle.log10(paddle.maximum(ones * amin, magnitude))
    log_spec -= 10.0 * math.log10(max(ref_value, amin))

    if top_db is not None:
        if top_db < 0:
            raise Exception("top_db must be non-negative")
        log_spec = paddle.maximum(log_spec, ones * (log_spec.max() - top_db))

    return log_spec


class Spectrogram(nn.Layer):
    def __init__(self,
                 n_fft: int=512,
                 hop_length: Optional[int]=None,
                 win_length: Optional[int]=None,
                 window: str='hann',
                 center: bool=True,
                 pad_mode: str='reflect',
                 dtype: str=paddle.float32):
        """Compute spectrogram of a given signal, typically an audio waveform.
        The spectorgram is defined as the complex norm of the short-time
        Fourier transformation.
        Parameters:
            n_fft(int): the number of frequency components of the discrete Fourier transform.
                The default value is 2048,
            hop_length(int|None): the hop length of the short time FFT. If None, it is set to win_length//4.
                The default value is None.
            win_length: the window length of the short time FFt. If None, it is set to same as n_fft.
                The default value is None.
            window(str): the name of the window function applied to the single before the Fourier transform.
                The folllowing window names are supported: 'hamming','hann','kaiser','gaussian',
                'exponential','triang','bohman','blackman','cosine','tukey','taylor'.
                The default value is 'hann'
            center(bool): if True, the signal is padded so that frame t is centered at x[t * hop_length].
                If False, frame t begins at x[t * hop_length]
                The default value is True
            pad_mode(str): the mode to pad the signal if necessary. The supported modes are 'reflect'
                and 'constant'. The default value is 'reflect'.
            dtype(str): the data type of input and window.
        Notes:
            The Spectrogram transform relies on STFT transform to compute the spectrogram.
            By default, the weights are not learnable. To fine-tune the Fourier coefficients,
            set stop_gradient=False before training.
            For more information, see STFT().
        """
        super(Spectrogram, self).__init__()

        if win_length is None:
            win_length = n_fft

        fft_window = get_window(window, win_length, fftbins=True, dtype=dtype)
        self._stft = partial(
            paddle.signal.stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=fft_window,
            center=center,
            pad_mode=pad_mode)

    def forward(self, x):
        stft = self._stft(x)
        spectrogram = paddle.square(paddle.abs(stft))
        return spectrogram


class MelSpectrogram(nn.Layer):
    def __init__(self,
                 sr: int=22050,
                 n_fft: int=512,
                 hop_length: Optional[int]=None,
                 win_length: Optional[int]=None,
                 window: str='hann',
                 center: bool=True,
                 pad_mode: str='reflect',
                 n_mels: int=64,
                 f_min: float=50.0,
                 f_max: Optional[float]=None,
                 htk: bool=False,
                 norm: Union[str, float]='slaney',
                 dtype: str=paddle.float32):
        """Compute the melspectrogram of a given signal, typically an audio waveform.
        The melspectrogram is also known as filterbank or fbank feature in audio community.
        It is computed by multiplying spectrogram with Mel filter bank matrix.
        Parameters:
            sr(int): the audio sample rate.
                The default value is 22050.
            n_fft(int): the number of frequency components of the discrete Fourier transform.
                The default value is 2048,
            hop_length(int|None): the hop length of the short time FFT. If None, it is set to win_length//4.
                The default value is None.
            win_length: the window length of the short time FFt. If None, it is set to same as n_fft.
                The default value is None.
            window(str): the name of the window function applied to the single before the Fourier transform.
                The folllowing window names are supported: 'hamming','hann','kaiser','gaussian',
                'exponential','triang','bohman','blackman','cosine','tukey','taylor'.
                The default value is 'hann'
            center(bool): if True, the signal is padded so that frame t is centered at x[t * hop_length].
                If False, frame t begins at x[t * hop_length]
                The default value is True
            pad_mode(str): the mode to pad the signal if necessary. The supported modes are 'reflect'
                and 'constant'.
                The default value is 'reflect'.
            n_mels(int): the mel bins.
            f_min(float): the lower cut-off frequency, below which the filter response is zero.
            f_max(float): the upper cut-off frequency, above which the filter response is zeros.
            htk(bool): whether to use HTK formula in computing fbank matrix.
            norm(str|float): the normalization type in computing fbank matrix.  Slaney-style is used by default.
                You can specify norm=1.0/2.0 to use customized p-norm normalization.
            dtype(str): the datatype of fbank matrix used in the transform. Use float64 to increase numerical
                accuracy. Note that the final transform will be conducted in float32 regardless of dtype of fbank matrix.
        """
        super(MelSpectrogram, self).__init__()

        self._spectrogram = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            dtype=dtype)
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.htk = htk
        self.norm = norm
        if f_max is None:
            f_max = sr // 2
        self.fbank_matrix = compute_fbank_matrix(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            htk=htk,
            norm=norm,
            dtype=dtype)  # float64 for better numerical results
        self.register_buffer('fbank_matrix', self.fbank_matrix)

    def forward(self, x):
        spect_feature = self._spectrogram(x)
        mel_feature = paddle.matmul(self.fbank_matrix, spect_feature)
        return mel_feature


class LogMelSpectrogram(nn.Layer):
    def __init__(self,
                 sr: int=22050,
                 n_fft: int=512,
                 hop_length: Optional[int]=None,
                 win_length: Optional[int]=None,
                 window: str='hann',
                 center: bool=True,
                 pad_mode: str='reflect',
                 n_mels: int=64,
                 f_min: float=50.0,
                 f_max: Optional[float]=None,
                 htk: bool=False,
                 norm: Union[str, float]='slaney',
                 ref_value: float=1.0,
                 amin: float=1e-10,
                 top_db: Optional[float]=None,
                 dtype: str=paddle.float32):
        """Compute log-mel-spectrogram(also known as LogFBank) feature of a given signal,
        typically an audio waveform.
        Parameters:
            sr(int): the audio sample rate.
                The default value is 22050.
            n_fft(int): the number of frequency components of the discrete Fourier transform.
                The default value is 2048,
            hop_length(int|None): the hop length of the short time FFT. If None, it is set to win_length//4.
                The default value is None.
            win_length: the window length of the short time FFt. If None, it is set to same as n_fft.
                The default value is None.
            window(str): the name of the window function applied to the single before the Fourier transform.
                The folllowing window names are supported: 'hamming','hann','kaiser','gaussian',
                'exponential','triang','bohman','blackman','cosine','tukey','taylor'.
                The default value is 'hann'
            center(bool): if True, the signal is padded so that frame t is centered at x[t * hop_length].
                If False, frame t begins at x[t * hop_length]
                The default value is True
            pad_mode(str): the mode to pad the signal if necessary. The supported modes are 'reflect'
                and 'constant'.
                The default value is 'reflect'.
            n_mels(int): the mel bins.
            f_min(float): the lower cut-off frequency, below which the filter response is zero.
            f_max(float): the upper cut-off frequency, above which the filter response is zeros.
            ref_value(float): the reference value. If smaller than 1.0, the db level
            htk(bool): whether to use HTK formula in computing fbank matrix.
            norm(str|float): the normalization type in computing fbank matrix. Slaney-style is used by default.
                You can specify norm=1.0/2.0 to use customized p-norm normalization.
            dtype(str): the datatype of fbank matrix used in the transform. Use float64 to increase numerical
                accuracy. Note that the final transform will be conducted in float32 regardless of dtype of fbank matrix.
            amin(float): the minimum value of input magnitude, below which the input of the signal will be pulled up accordingly.
                Otherwise, the db level is pushed down.
                magnitude is clipped(to amin). For numerical stability, set amin to a larger value,
                e.g., 1e-3.
            top_db(float): the maximum db value of resulting spectrum, above which the
                spectrum is clipped(to top_db).
        """
        super(LogMelSpectrogram, self).__init__()

        self._melspectrogram = MelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            htk=htk,
            norm=norm,
            dtype=dtype)

        self.ref_value = ref_value
        self.amin = amin
        self.top_db = top_db

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        mel_feature = self._melspectrogram(x)
        log_mel_feature = power_to_db(
            mel_feature,
            ref_value=self.ref_value,
            amin=self.amin,
            top_db=self.top_db)
        return log_mel_feature
