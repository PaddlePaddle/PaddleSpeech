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
# Modified from espnet(https://github.com/espnet/espnet)
import paddle
from paddle import nn
from paddle.nn import functional as F
from scipy import signal


def stft(x,
         fft_size,
         hop_length=None,
         win_length=None,
         window='hann',
         center=True,
         pad_mode='reflect'):
    """Perform STFT and convert to magnitude spectrogram.
    Parameters
    ----------
    x : Tensor
        Input signal tensor (B, T).
    fft_size : int
        FFT size.
    hop_size : int
        Hop size.
    win_length : int
        window : str, optional
    window : str
        Name of window function, see `scipy.signal.get_window` for more
        details. Defaults to "hann".
    center : bool, optional
        center (bool, optional): Whether to pad `x` to make that the
        :math:`t \times hop\_length` at the center of :math:`t`-th frame. Default: `True`.
    pad_mode : str, optional
        Choose padding pattern when `center` is `True`.
    Returns
    ----------
    Tensor:
        Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    # calculate window
    window = signal.get_window(window, win_length, fftbins=True)
    window = paddle.to_tensor(window)
    x_stft = paddle.signal.stft(
        x,
        fft_size,
        hop_length,
        win_length,
        window=window,
        center=center,
        pad_mode=pad_mode)

    real = x_stft.real()
    imag = x_stft.imag()

    return paddle.sqrt(paddle.clip(real**2 + imag**2, min=1e-7)).transpose(
        [0, 2, 1])


class SpectralConvergenceLoss(nn.Layer):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super().__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Parameters
        ----------
        x_mag : Tensor
            Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        y_mag : Tensor)
            Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns
        ----------
        Tensor
            Spectral convergence loss value.
        """
        return paddle.norm(
            y_mag - x_mag, p="fro") / paddle.clip(
                paddle.norm(y_mag, p="fro"), min=1e-10)


class LogSTFTMagnitudeLoss(nn.Layer):
    """Log STFT magnitude loss module."""

    def __init__(self, epsilon=1e-7):
        """Initilize los STFT magnitude loss module."""
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Parameters
        ----------
        x_mag : Tensor
            Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        y_mag : Tensor
            Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns
        ----------
        Tensor
            Log STFT magnitude loss value.
        """
        return F.l1_loss(
            paddle.log(paddle.clip(y_mag, min=self.epsilon)),
            paddle.log(paddle.clip(x_mag, min=self.epsilon)))


class STFTLoss(nn.Layer):
    """STFT loss module."""

    def __init__(self,
                 fft_size=1024,
                 shift_size=120,
                 win_length=600,
                 window="hann"):
        """Initialize STFT loss module."""
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = window
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Parameters
        ----------
        x : Tensor
            Predicted signal (B, T).
        y : Tensor
            Groundtruth signal (B, T).
        Returns
        ----------
        Tensor
            Spectral convergence loss value.
        Tensor
            Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length,
                     self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length,
                     self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Layer):
    """Multi resolution STFT loss module."""

    def __init__(
            self,
            fft_sizes=[1024, 2048, 512],
            hop_sizes=[120, 240, 50],
            win_lengths=[600, 1200, 240],
            window="hann", ):
        """Initialize Multi resolution STFT loss module.
        Parameters
        ----------
        fft_sizes : list
            List of FFT sizes.
        hop_sizes : list
            List of hop sizes.
        win_lengths : list
            List of window lengths.
        window : str
            Window function type.
        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.LayerList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, ss, wl, window))

    def forward(self, x, y):
        """Calculate forward propagation.
        Parameters
        ----------
        x : Tensor
            Predicted signal (B, T) or (B, #subband, T).
        y : Tensor
            Groundtruth signal (B, T) or (B, #subband, T).
        Returns
        ----------
        Tensor
            Multi resolution spectral convergence loss value.
        Tensor
            Multi resolution log STFT magnitude loss value.
        """
        if len(x.shape) == 3:
            # (B, C, T) -> (B x C, T)
            x = x.reshape([-1, x.shape[2]])
            # (B, C, T) -> (B x C, T)
            y = y.reshape([-1, y.shape[2]])
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
