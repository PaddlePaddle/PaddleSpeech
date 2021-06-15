from typing import Tuple
import numpy as np
import paddle
from paddle import Tensor
from paddle import nn
from paddle.nn import functional as F
import soundfile as sf

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


def _povey_window(frame_len:int) -> np.ndarray:
    win = np.empty(frame_len)
    for i in range(frame_len):
        win[i] = (0.5 - 0.5 * np.cos(2 * np.pi * i / (frame_len - 1)) )**0.85 
    return win


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

        # https://github.com/numpy/numpy/blob/v1.20.0/numpy/fft/_pocketfft.py#L49
        kernel_size = min(self.n_fft, self.win_length)
        
        # calculate window
        if not window_type:
            window = np.ones(kernel_size)
        elif window_type == "hann":
            window = np.hanning(kernel_size)
        elif window_type == "hamm":
            window = np.hamming(kernel_size)
        elif window_type == "povey":
            window = _povey_window(kernel_size)
        else:
            msg = f"{window_type} Not supported yet!"
            raise ValueError(msg)

        # https://en.wikipedia.org/wiki/Discrete_Fourier_transform
        # (n_bins, n_fft) complex
        n = np.arange(0, self.n_fft, 1.)
        wsin = np.empty((self.n_bin, kernel_size)) #[Cout, kernel_size]
        wcos = np.empty((self.n_bin, kernel_size)) #[Cout, kernel_size]
        for k in range(self.n_bin): # Only half of the bins contain useful info
            wsin[k,:] = np.sin(2*np.pi*k*n/self.n_fft)[:kernel_size]
            wcos[k,:] = np.cos(2*np.pi*k*n/self.n_fft)[:kernel_size]
        w_real = wcos
        w_imag = wsin
        
        # https://en.wikipedia.org/wiki/DFT_matrix
        # https://ccrma.stanford.edu/~jos/st/Matrix_Formulation_DFT.html
        # weight = np.fft.fft(np.eye(n_fft))[:self.n_bin, :kernel_size]
        # w_real = weight.real
        # w_imag = weight.imag

        # (2 * n_bins, kernel_size)
        #w = np.concatenate([w_real, w_imag], axis=0)
        w = w_real
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
        D : Tensor
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
        D = F.conv1d(x, self.weight,
                     stride=(self.stride_length, ),
                     padding=padding,
                     data_format="NLC")
        #D = paddle.reshape(D, [batch_size, -1, self.n_bin, 2])
        D = paddle.reshape(D, [batch_size, -1, self.n_bin, 1])
        return D, num_frames

