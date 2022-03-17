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
from functools import partial
from typing import Optional
from typing import Union

import paddle
import paddle.nn as nn

from ..functional import compute_fbank_matrix
from ..functional import create_dct
from ..functional import power_to_db
from ..functional.window import get_window

__all__ = [
    'Spectrogram',
    'MelSpectrogram',
    'LogMelSpectrogram',
    'MFCC',
]


class Spectrogram(nn.Layer):
    def __init__(self,
                 n_fft: int=512,
                 hop_length: Optional[int]=None,
                 win_length: Optional[int]=None,
                 window: str='hann',
                 power: float=2.0,
                 center: bool=True,
                 pad_mode: str='reflect',
                 dtype: str=paddle.float32):
        """Compute spectrogram of a given signal, typically an audio waveform.
        The spectorgram is defined as the complex norm of the short-time
        Fourier transformation.

        Args:
            n_fft (int, optional): The number of frequency components of the discrete Fourier transform. Defaults to 512.
            hop_length (Optional[int], optional): The hop length of the short time FFT. If `None`, it is set to `win_length//4`. Defaults to None.
            win_length (Optional[int], optional): The window length of the short time FFT. If `None`, it is set to same as `n_fft`. Defaults to None.
            window (str, optional): The window function applied to the single before the Fourier transform. Supported window functions: 'hamming', 'hann', 'kaiser', 'gaussian', 'exponential', 'triang', 'bohman', 'blackman', 'cosine', 'tukey', 'taylor'. Defaults to 'hann'.
            power (float, optional): Exponent for the magnitude spectrogram. Defaults to 2.0.
            center (bool, optional): Whether to pad `x` to make that the :math:`t \times hop\_length` at the center of `t`-th frame. Defaults to True.
            pad_mode (str, optional): Choose padding pattern when `center` is `True`. Defaults to 'reflect'.
            dtype (str, optional): Data type of input and window. Defaults to paddle.float32.
        """
        super(Spectrogram, self).__init__()

        assert power > 0, 'Power of spectrogram must be > 0.'
        self.power = power

        if win_length is None:
            win_length = n_fft

        self.fft_window = get_window(
            window, win_length, fftbins=True, dtype=dtype)
        self._stft = partial(
            paddle.signal.stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.fft_window,
            center=center,
            pad_mode=pad_mode)
        self.register_buffer('fft_window', self.fft_window)

    def forward(self, x):
        stft = self._stft(x)
        spectrogram = paddle.pow(paddle.abs(stft), self.power)
        return spectrogram


class MelSpectrogram(nn.Layer):
    def __init__(self,
                 sr: int=22050,
                 n_fft: int=512,
                 hop_length: Optional[int]=None,
                 win_length: Optional[int]=None,
                 window: str='hann',
                 power: float=2.0,
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
            power (float): Exponent for the magnitude spectrogram. The default value is 2.0.
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
            power=power,
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
                 power: float=2.0,
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
            sr (int): the audio sample rate.
                The default value is 22050.
            n_fft (int): the number of frequency components of the discrete Fourier transform.
                The default value is 2048,
            hop_length (int|None): the hop length of the short time FFT. If None, it is set to win_length//4.
                The default value is None.
            win_length: the window length of the short time FFt. If None, it is set to same as n_fft.
                The default value is None.
            window (str): the name of the window function applied to the single before the Fourier transform.
                The folllowing window names are supported: 'hamming','hann','kaiser','gaussian',
                'exponential','triang','bohman','blackman','cosine','tukey','taylor'.
                The default value is 'hann'
            center (bool): if True, the signal is padded so that frame t is centered at x[t * hop_length].
                If False, frame t begins at x[t * hop_length]
                The default value is True
            pad_mode (str): the mode to pad the signal if necessary. The supported modes are 'reflect'
                and 'constant'.
                The default value is 'reflect'.
            n_mels (int): the mel bins.
            f_min (float): the lower cut-off frequency, below which the filter response is zero.
            f_max (float): the upper cut-off frequency, above which the filter response is zeros.
            htk (bool): whether to use HTK formula in computing fbank matrix.
            norm (str|float): the normalization type in computing fbank matrix. Slaney-style is used by default.
                You can specify norm=1.0/2.0 to use customized p-norm normalization.
            ref_value (float): the reference value. If smaller than 1.0, the db level of the signal will be pulled up accordingly. Otherwise, the db level is pushed down.
            amin (float): the minimum value of input magnitude, below which the input magnitude is clipped(to amin).
            top_db (float): the maximum db value of resulting spectrum, above which the
                spectrum is clipped(to top_db).
            dtype (str): the datatype of fbank matrix used in the transform. Use float64 to increase numerical
                accuracy. Note that the final transform will be conducted in float32 regardless of dtype of fbank matrix.
        """
        super(LogMelSpectrogram, self).__init__()

        self._melspectrogram = MelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            power=power,
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
        mel_feature = self._melspectrogram(x)
        log_mel_feature = power_to_db(
            mel_feature,
            ref_value=self.ref_value,
            amin=self.amin,
            top_db=self.top_db)
        return log_mel_feature


class MFCC(nn.Layer):
    def __init__(self,
                 sr: int=22050,
                 n_mfcc: int=40,
                 n_fft: int=512,
                 hop_length: Optional[int]=None,
                 win_length: Optional[int]=None,
                 window: str='hann',
                 power: float=2.0,
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
        """Compute mel frequency cepstral coefficients(MFCCs) feature of given waveforms.

        Parameters:
            sr(int): the audio sample rate.
                The default value is 22050.
            n_mfcc (int, optional): Number of cepstra in MFCC. Defaults to 40.
            n_fft (int): the number of frequency components of the discrete Fourier transform.
                The default value is 2048,
            hop_length (int|None): the hop length of the short time FFT. If None, it is set to win_length//4.
                The default value is None.
            win_length: the window length of the short time FFt. If None, it is set to same as n_fft.
                The default value is None.
            window (str): the name of the window function applied to the single before the Fourier transform.
                The folllowing window names are supported: 'hamming','hann','kaiser','gaussian',
                'exponential','triang','bohman','blackman','cosine','tukey','taylor'.
                The default value is 'hann'
            power (float): Exponent for the magnitude spectrogram. The default value is 2.0.
            center (bool): if True, the signal is padded so that frame t is centered at x[t * hop_length].
                If False, frame t begins at x[t * hop_length]
                The default value is True
            pad_mode (str): the mode to pad the signal if necessary. The supported modes are 'reflect'
                and 'constant'.
                The default value is 'reflect'.
            n_mels (int): the mel bins.
            f_min (float): the lower cut-off frequency, below which the filter response is zero.
            f_max (float): the upper cut-off frequency, above which the filter response is zeros.
            htk (bool): whether to use HTK formula in computing fbank matrix.
            norm (str|float): the normalization type in computing fbank matrix. Slaney-style is used by default.
                You can specify norm=1.0/2.0 to use customized p-norm normalization.
            ref_value (float): the reference value. If smaller than 1.0, the db level of the signal will be pulled up accordingly. Otherwise, the db level is pushed down.
            amin (float): the minimum value of input magnitude, below which the input magnitude is clipped(to amin).
            top_db (float): the maximum db value of resulting spectrum, above which the
                spectrum is clipped(to top_db).
            dtype (str): the datatype of fbank matrix used in the transform. Use float64 to increase numerical
                accuracy. Note that the final transform will be conducted in float32 regardless of dtype of fbank matrix.
        """
        super(MFCC, self).__init__()
        assert n_mfcc <= n_mels, 'n_mfcc cannot be larger than n_mels: %d vs %d' % (
            n_mfcc, n_mels)
        self._log_melspectrogram = LogMelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            power=power,
            center=center,
            pad_mode=pad_mode,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            htk=htk,
            norm=norm,
            ref_value=ref_value,
            amin=amin,
            top_db=top_db,
            dtype=dtype)
        self.dct_matrix = create_dct(n_mfcc=n_mfcc, n_mels=n_mels, dtype=dtype)
        self.register_buffer('dct_matrix', self.dct_matrix)

    def forward(self, x):
        log_mel_feature = self._log_melspectrogram(x)
        mfcc = paddle.matmul(
            log_mel_feature.transpose((0, 2, 1)), self.dct_matrix).transpose(
                (0, 2, 1))  # (B, n_mels, L)
        return mfcc
