# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import nn
from paddle.fluid.layers import sequence_mask
from paddle.nn import functional as F
from scipy import signal


# Loss for Tacotron2
def attention_guide(dec_lens, enc_lens, N, T, g, dtype=None):
    """Build that W matrix. shape(B, T_dec, T_enc)
    W[i, n, t] = 1 - exp(-(n/dec_lens[i] - t/enc_lens[i])**2 / (2g**2)) 

    See also:
    Tachibana, Hideyuki, Katsuya Uenoyama, and Shunsuke Aihara. 2017. “Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.” ArXiv:1710.08969 [Cs, Eess], October. http://arxiv.org/abs/1710.08969.
    """
    dtype = dtype or paddle.get_default_dtype()
    dec_pos = paddle.arange(0, N).astype(dtype) / dec_lens.unsqueeze(
        -1)  # n/N # shape(B, T_dec)
    enc_pos = paddle.arange(0, T).astype(dtype) / enc_lens.unsqueeze(
        -1)  # t/T # shape(B, T_enc)
    W = 1 - paddle.exp(-(dec_pos.unsqueeze(-1) - enc_pos.unsqueeze(1))**2 /
                       (2 * g**2))

    dec_mask = sequence_mask(dec_lens, maxlen=N)
    enc_mask = sequence_mask(enc_lens, maxlen=T)
    mask = dec_mask.unsqueeze(-1) * enc_mask.unsqueeze(1)
    mask = paddle.cast(mask, W.dtype)

    W *= mask
    return W


def guided_attention_loss(attention_weight, dec_lens, enc_lens, g):
    """Guided attention loss, masked to excluded padding parts."""
    _, N, T = attention_weight.shape
    W = attention_guide(dec_lens, enc_lens, N, T, g, attention_weight.dtype)

    total_tokens = (dec_lens * enc_lens).astype(W.dtype)
    loss = paddle.mean(paddle.sum(W * attention_weight, [1, 2]) / total_tokens)
    return loss


# Losses for GAN Vocoder
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


class GeneratorAdversarialLoss(nn.Layer):
    """Generator adversarial loss module."""

    def __init__(
            self,
            average_by_discriminators=True,
            loss_type="mse", ):
        """Initialize GeneratorAversarialLoss module."""
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.criterion = self._mse_loss
        else:
            self.criterion = self._hinge_loss

    def forward(self, outputs):
        """Calcualate generator adversarial loss.
        Parameters
        ----------
        outputs: Tensor or List
        Discriminator outputs or list of discriminator outputs.
        Returns
        ----------
        Tensor
            Generator adversarial loss value.
        """
        if isinstance(outputs, (tuple, list)):
            adv_loss = 0.0
            for i, outputs_ in enumerate(outputs):
                if isinstance(outputs_, (tuple, list)):
                    # case including feature maps
                    outputs_ = outputs_[-1]
                adv_loss += self.criterion(outputs_)
            if self.average_by_discriminators:
                adv_loss /= i + 1
        else:
            adv_loss = self.criterion(outputs)

        return adv_loss

    def _mse_loss(self, x):
        return F.mse_loss(x, paddle.ones_like(x))

    def _hinge_loss(self, x):
        return -x.mean()


class DiscriminatorAdversarialLoss(nn.Layer):
    """Discriminator adversarial loss module."""

    def __init__(
            self,
            average_by_discriminators=True,
            loss_type="mse", ):
        """Initialize DiscriminatorAversarialLoss module."""
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss

    def forward(self, outputs_hat, outputs):
        """Calcualate discriminator adversarial loss.
        Parameters
        ----------
        outputs_hat : Tensor or list
            Discriminator outputs or list of
            discriminator outputs calculated from generator outputs.
        outputs : Tensor or list
            Discriminator outputs or list of
            discriminator outputs calculated from groundtruth.
        Returns
        ----------
        Tensor
            Discriminator real loss value.
        Tensor
            Discriminator fake loss value.
        """
        if isinstance(outputs, (tuple, list)):
            real_loss = 0.0
            fake_loss = 0.0
            for i, (outputs_hat_,
                    outputs_) in enumerate(zip(outputs_hat, outputs)):
                if isinstance(outputs_hat_, (tuple, list)):
                    # case including feature maps
                    outputs_hat_ = outputs_hat_[-1]
                    outputs_ = outputs_[-1]
                real_loss += self.real_criterion(outputs_)
                fake_loss += self.fake_criterion(outputs_hat_)
            if self.average_by_discriminators:
                fake_loss /= i + 1
                real_loss /= i + 1
        else:
            real_loss = self.real_criterion(outputs)
            fake_loss = self.fake_criterion(outputs_hat)

        return real_loss, fake_loss

    def _mse_real_loss(self, x):
        return F.mse_loss(x, paddle.ones_like(x))

    def _mse_fake_loss(self, x):
        return F.mse_loss(x, paddle.zeros_like(x))


# Losses for SpeedySpeech
# Structural Similarity Index Measure (SSIM)
def gaussian(window_size, sigma):
    gauss = paddle.to_tensor([
        math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = paddle.matmul(_1D_window, paddle.transpose(
        _1D_window, [1, 0])).unsqueeze([0, 1])
    window = paddle.expand(_2D_window, [channel, 1, window_size, window_size])
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) \
             / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.shape
    window = create_window(window_size, channel)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def weighted_mean(input, weight):
    """Weighted mean. It can also be used as masked mean.

    Parameters
    -----------
    input : Tensor 
        The input tensor.
    weight : Tensor
        The weight tensor with broadcastable shape with the input.

    Returns
    ----------
    Tensor [shape=(1,)]
        Weighted mean tensor with the same dtype as input.
    """
    weight = paddle.cast(weight, input.dtype)
    broadcast_ratio = input.size / weight.size
    return paddle.sum(input * weight) / (paddle.sum(weight) * broadcast_ratio)


def masked_l1_loss(prediction, target, mask):
    """Compute maksed L1 loss.

    Parameters
    ----------
    prediction : Tensor
        The prediction.
    target : Tensor
        The target. The shape should be broadcastable to ``prediction``.
    mask : Tensor
        The mask. The shape should be broadcatable to the broadcasted shape of
        ``prediction`` and ``target``.

    Returns
    -------
    Tensor [shape=(1,)]
        The masked L1 loss.
    """
    abs_error = F.l1_loss(prediction, target, reduction='none')
    loss = weighted_mean(abs_error, mask)
    return loss
