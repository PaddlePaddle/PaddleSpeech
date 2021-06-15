from typing import Tuple
import numpy as np
import paddle
from paddle import Tensor
from paddle import nn
from paddle.nn import functional as F
import soundfile as sf

from .common import get_window, dft_matrix


def read(wavpath:str, sr:int = None, dtype='int16')->Tuple[int, np.ndarray]:
    wav, r_sr = sf.read(wavpath, dtype=dtype)
    if sr:
        assert sr == r_sr
    return r_sr, wav

 
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
                 window_type: str = None,
                 clip: bool = False):
        super().__init__()
        self.sr = sr
        self.win_length = int(win_length * sr)
        self.stride_length = int(stride_length * sr)
        self.clip = clip
        
        self.n_fft = n_fft
        self.n_bin = 1 + n_fft // 2

        w_real, w_imag, kernel_size = dft_matrix(self.n_fft, self.win_length, self.n_bin)
        
        # calculate window
        window = get_window(window_type, kernel_size)

        # (2 * n_bins, kernel_size)
        w = np.concatenate([w_real, w_imag], axis=0)
        w = w * window

        # (2 * n_bins, 1, kernel_size) # (C_out, C_in, kernel_size)
        w = np.expand_dims(w, 1)
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
        num_frames = (num_samples - self.win_length) // self.stride_length
        padding = (0, 0)
        if not self.clip:
            num_frames += 1
            need_samples = num_frames * self.stride_length + self.win_length
            padding = (0, need_samples - num_samples - 1)

        batch_size, _ = paddle.shape(x)
        x = x.unsqueeze(-1)
        C = F.conv1d(x, self.weight,
                     stride=(self.stride_length, ),
                     padding=padding,
                     data_format="NLC")
        C = paddle.reshape(C, [batch_size, -1, 2, self.n_bin])
        C = C.transpose([0, 1, 3, 2])
        return C, num_frames


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


