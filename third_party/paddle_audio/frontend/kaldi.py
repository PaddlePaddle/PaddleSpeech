from typing import Tuple
import numpy as np
import paddle
from paddle import Tensor
from paddle import nn
from paddle.nn import functional as F
import soundfile as sf

from .common import get_window
from .common import dft_matrix


def read(wavpath:str, sr:int = None, start=0, stop=None, dtype='int16', always_2d=True)->Tuple[int, np.ndarray]:
    """load wav file.

    Args:
        wavpath (str): wav path.
        sr (int, optional): expect sample rate. Defaults to None.
        dtype (str, optional): wav data bits. Defaults to 'int16'.

    Returns:
        Tuple[int, np.ndarray]: sr (int), wav (int16) [T, C].
    """
    wav, r_sr = sf.read(wavpath, start=start, stop=stop, dtype=dtype, always_2d=always_2d)
    if sr:
        assert sr == r_sr
    return r_sr, wav


def write(wavpath:str, wav:np.ndarray, sr:int, dtype='PCM_16'):
    """write wav file.

    Args:
        wavpath (str): file path to save.
        wav (np.ndarray): wav data.
        sr (int): data samplerate.
        dtype (str, optional): wav bit format. Defaults to 'PCM_16'.
    """
    sf.write(wavpath, wav, sr, subtype=dtype)


def frames(x: Tensor,
          num_samples: Tensor,
          sr: int,
          win_length: float,
          stride_length: float,
          clip: bool = False) -> Tuple[Tensor, Tensor]:
    """Extract frames from audio.

    Parameters
    ----------
    x : Tensor
        Shape (B, T), batched waveform.
    num_samples : Tensor
        Shape (B, ), number of samples of each waveform.
    sr: int
        Sampling Rate.
    win_length : float
        Window length in ms.
    stride_length : float
        Stride length in ms.
    clip : bool, optional
        Whether to clip audio that does not fit into the last frame, by
        default True

    Returns
    -------
    frames : Tensor
        Shape (B, T', win_length).
    num_frames : Tensor
        Shape (B, ) number of valid frames
    """
    assert stride_length <= win_length
    stride_length = int(stride_length * sr)
    win_length = int(win_length * sr)

    num_frames = (num_samples - win_length) // stride_length
    padding = (0, 0)
    if not clip:
        num_frames += 1
        need_samples = num_frames * stride_length + win_length
        padding = (0, need_samples - num_samples - 1)

    weight = paddle.eye(win_length).unsqueeze(1) #[win_length, 1, win_length]

    frames = F.conv1d(x.unsqueeze(-1),
                      weight,
                      padding=padding,
                      stride=(stride_length, ),
                      data_format='NLC')
    return frames, num_frames


def dither(signal:Tensor, dither_value=1.0)->Tensor:
    """dither frames for log compute.

    Args:
        signal (Tensor): [B, T, D]
        dither_value (float, optional): [scalar]. Defaults to 1.0.

    Returns:
        Tensor: [B, T, D]
    """
    D = paddle.shape(signal)[-1]
    signal += paddle.normal(shape=[1, 1, D]) * dither_value
    return signal


def remove_dc_offset(signal:Tensor)->Tensor:
    """remove dc.

    Args:
        signal (Tensor): [B, T, D]

    Returns:
        Tensor: [B, T, D]
    """
    signal -= paddle.mean(signal, axis=-1, keepdim=True)
    return signal

def preemphasis(signal:Tensor, coeff=0.97)->Tensor:
    """perform preemphasis on the input signal.

    Args:
        signal (Tensor): [B, T, D], The signal to filter.
        coeff (float, optional): [scalar].The preemphasis coefficient. 0 is no filter, Defaults to 0.97.

    Returns:
        Tensor: [B, T, D]
    """
    return paddle.concat([
        (1-coeff)*signal[:, :, 0:1],
        signal[:, :, 1:] - coeff * signal[:, :, :-1]
    ], axis=-1)


class STFT(nn.Layer):
    """A module for computing stft transformation in a differentiable way.

    http://practicalcryptography.com/miscellaneous/machine-learning/intuitive-guide-discrete-fourier-transform/

    Parameters
    ------------
    n_fft : int
        Number of samples in a frame.

    sr: int
        Number of Samplilng rate.

    stride_length : float
        Number of samples shifted between adjacent frames.

    win_length : float
        Length of the window.

    clip: bool
        Whether to clip audio is necesaary.
    """
    def __init__(self,
                 n_fft: int,
                 sr: int,
                 win_length: float,
                 stride_length: float,
                 dither:float=0.0,
                 preemph_coeff:float=0.97,
                 remove_dc_offset:bool=True,
                 window_type: str = 'povey',
                 clip: bool = False):
        super().__init__()
        self.sr = sr
        self.win_length = win_length
        self.stride_length = stride_length
        self.dither = dither
        self.preemph_coeff = preemph_coeff
        self.remove_dc_offset = remove_dc_offset
        self.window_type = window_type
        self.clip = clip

        self.n_fft = n_fft
        self.n_bin = 1 + n_fft // 2

        w_real, w_imag, kernel_size = dft_matrix(
            self.n_fft, int(self.win_length * self.sr), self.n_bin
        )

        # calculate window
        window = get_window(window_type, kernel_size)

        # (2 * n_bins, kernel_size)
        w = np.concatenate([w_real, w_imag], axis=0)
        w = w * window
        # (kernel_size, 2 * n_bins)
        w = np.transpose(w)
        weight = paddle.cast(paddle.to_tensor(w), paddle.get_default_dtype())
        self.register_buffer("weight", weight)

    def forward(self, x: Tensor, num_samples: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the stft transform.
        Parameters
        ------------
        x : Tensor [shape=(B, T)]
            The input waveform.
        num_samples : Tensor [shape=(B,)]
            Number of samples of each waveform.
        Returns
        ------------
        C : Tensor
            Shape(B, T', n_bins, 2) Spectrogram.

        num_frames: Tensor
            Shape (B,) number of samples of each spectrogram
        """
        batch_size = paddle.shape(num_samples)
        F, nframe = frames(x, num_samples, self.sr, self.win_length, self.stride_length, clip=self.clip)
        if self.dither:
            F = dither(F, self.dither)
        if self.remove_dc_offset:
            F = remove_dc_offset(F)
        if self.preemph_coeff:
            F = preemphasis(F)
        C = paddle.matmul(F, self.weight) # [B, T, K] [K, 2 * n_bins]
        C = paddle.reshape(C, [batch_size, -1, 2, self.n_bin])
        C = C.transpose([0, 1, 3, 2])
        return C, nframe


def powspec(C:Tensor) -> Tensor:
    """Compute the power spectrum.

    Args:
        C (Tensor): [B, T, C, 2]

    Returns:
        Tensor: [B, T, C]
    """
    real, imag = paddle.chunk(C, 2, axis=-1)
    return paddle.square(real.squeeze(-1)) + paddle.square(imag.squeeze(-1))


def magspec(C: Tensor, eps=1e-10) -> Tensor:
    """Compute the magnitude spectrum.

    Args:
        C (Tensor): [B, T, C, 2]
        eps (float): epsilon.

    Returns:
        Tensor: [B, T, C]
    """
    pspec = powspec(C)
    return paddle.sqrt(pspec + eps)


