import paddle
import numpy as np
from typing import Tuple


# https://github.com/kaldi-asr/kaldi/blob/cbed4ff688/src/feat/feature-window.cc#L109
def povey_window(frame_len:int) -> np.ndarray:
    win = np.empty(frame_len)
    a = 2 * np.pi / (frame_len -1)
    for i in range(frame_len):
        win[i] = (0.5 - 0.5 * np.cos(a * i) )**0.85 
    return win

def hann_window(frame_len:int) -> np.ndarray:
    win = np.empty(frame_len)
    a = 2 * np.pi / (frame_len -1)
    for i in range(frame_len):
        win[i] = 0.5 - 0.5 * np.cos(a * i)
    return win

def sine_window(frame_len:int) -> np.ndarray:
    win = np.empty(frame_len)
    a = 2 * np.pi / (frame_len -1)
    for i in range(frame_len):
        win[i] = np.sin(0.5 * a * i)
    return win

def hamm_window(frame_len:int) -> np.ndarray:
    win = np.empty(frame_len)
    a = 2 * np.pi / (frame_len -1)
    for i in range(frame_len):
        win[i] = 0.54 - 0.46 * np.cos(a * i)
    return win

def get_window(wintype:str, winlen:int) -> np.ndarray:
    # calculate window
    if not wintype or wintype == 'rectangular':
        window = np.ones(winlen)
    elif wintype == "hann":
        window = hann_window(winlen)
    elif wintype == "hamm":
        window = hamm_window(winlen)
    elif wintype == "povey":
        window = povey_window(winlen)
    else:
        msg = f"{wintype} Not supported yet!"
        raise ValueError(msg)
    return window
    
   
def dft_matrix(n_fft:int, winlen:int=None, n_bin:int=None) -> Tuple[np.ndarray, np.ndarray, int]:
    # https://en.wikipedia.org/wiki/Discrete_Fourier_transform
    # (n_bins, n_fft) complex
    if n_bin is None:
        n_bin = 1 + n_fft // 2
    if winlen is None:
        winlen = n_bin
    # https://github.com/numpy/numpy/blob/v1.20.0/numpy/fft/_pocketfft.py#L49
    kernel_size = min(n_fft, winlen)
        
    n = np.arange(0, n_fft, 1.)
    wsin = np.empty((n_bin, kernel_size)) #[Cout, kernel_size]
    wcos = np.empty((n_bin, kernel_size)) #[Cout, kernel_size]
    for k in range(n_bin): # Only half of the bins contain useful info
        wsin[k,:] = -np.sin(2*np.pi*k*n/n_fft)[:kernel_size]
        wcos[k,:] = np.cos(2*np.pi*k*n/n_fft)[:kernel_size]
    w_real = wcos
    w_imag = wsin
    return w_real, w_imag, kernel_size
    

def dft_matrix_fast(n_fft:int, winlen:int=None, n_bin:int=None) -> Tuple[np.ndarray, np.ndarray, int]:
    # (n_bins, n_fft) complex
    if n_bin is None:
        n_bin = 1 + n_fft // 2
    if winlen is None:
        winlen = n_bin
    # https://github.com/numpy/numpy/blob/v1.20.0/numpy/fft/_pocketfft.py#L49
    kernel_size = min(n_fft, winlen)
    
    # https://en.wikipedia.org/wiki/DFT_matrix
    # https://ccrma.stanford.edu/~jos/st/Matrix_Formulation_DFT.html
    weight = np.fft.fft(np.eye(n_fft))[:self.n_bin, :kernel_size]
    w_real = weight.real
    w_imag = weight.imag
    return w_real, w_imag, kernel_size
        
        
def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 1127 * np.log(1+hz/700.0)

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (np.exp(mel/1127.0)-1)