from typing import Tuple
import numpy as np
import paddle
from paddle import Tensor
from paddle import nn
from paddle.nn import functional as F


def frame(x: Tensor,
          num_samples: Tensor,
          win_length: int,
          hop_length: int,
          clip: bool = True) -> Tuple[Tensor, Tensor]:
    """Extract frames from audio.

    Parameters
    ----------
    x : Tensor
        Shape (N, T), batched waveform.
    num_samples : Tensor
        Shape (N, ), number of samples of each waveform.
    win_length : int
        Window length.
    hop_length : int
        Number of samples shifted between ajancent frames.
    clip : bool, optional
        Whether to clip audio that does not fit into the last frame, by 
        default True

    Returns
    -------
    frames : Tensor
        Shape (N, T', win_length).
    num_frames : Tensor
        Shape (N, ) number of valid frames
    """
    assert hop_length <= win_length
    num_frames = (num_samples - win_length) // hop_length
    padding = (0, 0)
    if not clip:
        num_frames += 1
        # NOTE: pad hop_length - 1 to the right to ensure that there is at most
        # one frame dangling to the righe edge
        padding = (0, hop_length - 1)

    weight = paddle.eye(win_length).unsqueeze(1)

    frames = F.conv1d(x.unsqueeze(1),
                      weight,
                      padding=padding,
                      stride=(hop_length, ))
    return frames, num_frames


class STFT(nn.Layer):
    """A module for computing stft transformation in a differentiable way. 
    
    Parameters
    ------------
    n_fft : int
        Number of samples in a frame.
        
    hop_length : int
        Number of samples shifted between adjacent frames.
        
    win_length : int
        Length of the window.

    clip: bool
        Whether to clip audio is necesaary.
    """
    def __init__(self,
                 n_fft: int,
                 hop_length: int,
                 win_length: int,
                 window_type: str = None,
                 clip: bool = True):
        super().__init__()

        self.hop_length = hop_length
        self.n_bin = 1 + n_fft // 2
        self.n_fft = n_fft
        self.clip = clip

        # calculate window
        if window_type is None:
            window = np.ones(win_length)
        elif window_type == "hann":
            window = np.hanning(win_length)
        elif window_type == "hamming":
            window = np.hamming(win_length)
        else:
            raise ValueError("Not supported yet!")

        if win_length < n_fft:
            window = F.pad(window, (0, n_fft - win_length))
        elif win_length > n_fft:
            window = window[:n_fft]

        # (n_bins, n_fft) complex
        kernel_size = min(n_fft, win_length)
        weight = np.fft.fft(np.eye(n_fft))[:self.n_bin, :kernel_size]
        w_real = weight.real
        w_imag = weight.imag

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
        num_samples : Tensor 
            Number of samples of each waveform.
        Returns
        ------------
        D : Tensor
            Shape(N, T', n_bins, 2) Spectrogram.

        num_frames: Tensor
            Shape (N,) number of samples of each spectrogram
        """
        num_frames = (num_samples - self.win_length) // self.hop_length
        padding = (0, 0)
        if not self.clip:
            num_frames += 1
            padding = (0, self.hop_length - 1)

        batch_size, _, _ = paddle.shape(x)
        x = x.unsqueeze(-1)
        D = F.conv1d(self.weight,
                     x,
                     stride=(self.hop_length, ),
                     padding=padding,
                     data_format="NLC")
        D = paddle.reshape(D, [batch_size, -1, self.n_bin, 2])
        return D, num_frames

