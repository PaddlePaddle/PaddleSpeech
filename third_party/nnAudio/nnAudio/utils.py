"""
Module containing helper functions such as overlap sum and Fourier kernels generators
"""

import torch
from torch.nn.functional import conv1d, fold

import numpy as np
from time import time
import math
from scipy.signal import get_window
from scipy import signal
from scipy import fft
import warnings

from nnAudio.librosa_functions import * 

## --------------------------- Filter Design ---------------------------##
def torch_window_sumsquare(w, n_frames, stride, n_fft, power=2):
    w_stacks = w.unsqueeze(-1).repeat((1,n_frames)).unsqueeze(0)
    # Window length + stride*(frames-1)
    output_len = w_stacks.shape[1] + stride*(w_stacks.shape[2]-1) 
    return fold(w_stacks**power, (1,output_len), kernel_size=(1,n_fft), stride=stride)

def overlap_add(X, stride):
    n_fft = X.shape[1]
    output_len = n_fft + stride*(X.shape[2]-1) 
    
    return fold(X, (1,output_len), kernel_size=(1,n_fft), stride=stride).flatten(1)

def uniform_distribution(r1,r2, *size, device):
    return (r1 - r2) * torch.rand(*size, device=device) + r2

def extend_fbins(X):
    """Extending the number of frequency bins from `n_fft//2+1` back to `n_fft` by
       reversing all bins except DC and Nyquist and append it on top of existing spectrogram"""
    X_upper = torch.flip(X[:,1:-1],(0,1))
    X_upper[:,:,:,1] = -X_upper[:,:,:,1] # For the imaganinry part, it is an odd function
    return torch.cat((X[:, :, :], X_upper), 1)


def downsampling_by_n(x, filterKernel, n):
    """A helper function that downsamples the audio by a arbitary factor n.
    It is used in CQT2010 and CQT2010v2.

    Parameters
    ----------
    x : torch.Tensor
        The input waveform in ``torch.Tensor`` type with shape ``(batch, 1, len_audio)``

    filterKernel : str
        Filter kernel in ``torch.Tensor`` type with shape ``(1, 1, len_kernel)``

    n : int
        The downsampling factor

    Returns
    -------
    torch.Tensor
        The downsampled waveform

    Examples
    --------
    >>> x_down = downsampling_by_n(x, filterKernel)
    """

    x = conv1d(x,filterKernel,stride=n, padding=(filterKernel.shape[-1]-1)//2)
    return x


def downsampling_by_2(x, filterKernel):
    """A helper function that downsamples the audio by half. It is used in CQT2010 and CQT2010v2

    Parameters
    ----------
    x : torch.Tensor
        The input waveform in ``torch.Tensor`` type with shape ``(batch, 1, len_audio)``

    filterKernel : str
        Filter kernel in ``torch.Tensor`` type with shape ``(1, 1, len_kernel)``

    Returns
    -------
    torch.Tensor
        The downsampled waveform

    Examples
    --------
    >>> x_down = downsampling_by_2(x, filterKernel)
    """

    x = conv1d(x,filterKernel,stride=2, padding=(filterKernel.shape[-1]-1)//2)
    return x


## Basic tools for computation ##
def nextpow2(A):
    """A helper function to calculate the next nearest number to the power of 2.

    Parameters
    ----------
    A : float
        A float number that is going to be rounded up to the nearest power of 2

    Returns
    -------
    int
        The nearest power of 2 to the input number ``A``

    Examples
    --------

    >>> nextpow2(6)
    3
    """

    return int(np.ceil(np.log2(A)))

## Basic tools for computation ##
def prepow2(A):
    """A helper function to calculate the next nearest number to the power of 2.

    Parameters
    ----------
    A : float
        A float number that is going to be rounded up to the nearest power of 2

    Returns
    -------
    int
        The nearest power of 2 to the input number ``A``

    Examples
    --------

    >>> nextpow2(6)
    3
    """

    return int(np.floor(np.log2(A)))


def complex_mul(cqt_filter, stft):
    """Since PyTorch does not support complex numbers and its operation.
    We need to write our own complex multiplication function. This one is specially
    designed for CQT usage.

    Parameters
    ----------
    cqt_filter : tuple of torch.Tensor
        The tuple is in the format of ``(real_torch_tensor, imag_torch_tensor)``

    Returns
    -------
    tuple of torch.Tensor
        The output is in the format of ``(real_torch_tensor, imag_torch_tensor)``
    """

    cqt_filter_real = cqt_filter[0]
    cqt_filter_imag = cqt_filter[1]
    fourier_real = stft[0]
    fourier_imag = stft[1]

    CQT_real = torch.matmul(cqt_filter_real, fourier_real) - torch.matmul(cqt_filter_imag, fourier_imag)
    CQT_imag = torch.matmul(cqt_filter_real, fourier_imag) + torch.matmul(cqt_filter_imag, fourier_real)

    return CQT_real, CQT_imag


def broadcast_dim(x):
    """
    Auto broadcast input so that it can fits into a Conv1d
    """

    if x.dim() == 2:
        x = x[:, None, :]
    elif x.dim() == 1:
        # If nn.DataParallel is used, this broadcast doesn't work
        x = x[None, None, :]
    elif x.dim() == 3:
        pass
    else:
        raise ValueError("Only support input with shape = (batch, len) or shape = (len)")
    return x


def broadcast_dim_conv2d(x):
    """
    Auto broadcast input so that it can fits into a Conv2d
    """

    if x.dim() == 3:
        x = x[:, None, :,:]

    else:
        raise ValueError("Only support input with shape = (batch, len) or shape = (len)")
    return x


## Kernal generation functions ##
def create_fourier_kernels(n_fft, win_length=None, freq_bins=None, fmin=50,fmax=6000, sr=44100,
                           freq_scale='linear', window='hann', verbose=True):
    """ This function creates the Fourier Kernel for STFT, Melspectrogram and CQT.
    Most of the parameters follow librosa conventions. Part of the code comes from
    pytorch_musicnet. https://github.com/jthickstun/pytorch_musicnet

    Parameters
    ----------
    n_fft : int
        The window size

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins

    fmin : int
        The starting frequency for the lowest frequency bin.
        If freq_scale is ``no``, this argument does nothing.

    fmax : int
        The ending frequency for the highest frequency bin.
        If freq_scale is ``no``, this argument does nothing.

    sr : int
        The sampling rate for the input audio. It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    freq_scale: 'linear', 'log', or 'no'
        Determine the spacing between each frequency bin.
        When 'linear' or 'log' is used, the bin spacing can be controlled by ``fmin`` and ``fmax``.
        If 'no' is used, the bin will start at 0Hz and end at Nyquist frequency with linear spacing.

    Returns
    -------
    wsin : numpy.array
        Imaginary Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``

    wcos : numpy.array
        Real Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``

    bins2freq : list
        Mapping each frequency bin to frequency in Hz.

    binslist : list
        The normalized frequency ``k`` in digital domain.
        This ``k`` is in the Discrete Fourier Transform equation $$

    """

    if freq_bins==None: freq_bins = n_fft//2+1
    if win_length==None: win_length = n_fft

    s = np.arange(0, n_fft, 1.)
    wsin = np.empty((freq_bins,1,n_fft))
    wcos = np.empty((freq_bins,1,n_fft))
    start_freq = fmin
    end_freq = fmax
    bins2freq = []
    binslist = []

    # num_cycles = start_freq*d/44000.
    # scaling_ind = np.log(end_freq/start_freq)/k

    # Choosing window shape

    window_mask = get_window(window,int(win_length), fftbins=True)
    window_mask = pad_center(window_mask, n_fft)

    if freq_scale == 'linear':
        if verbose==True:
            print(f"sampling rate = {sr}. Please make sure the sampling rate is correct in order to"
                  f"get a valid freq range")
        start_bin = start_freq*n_fft/sr
        scaling_ind = (end_freq-start_freq)*(n_fft/sr)/freq_bins

        for k in range(freq_bins): # Only half of the bins contain useful info
            # print("linear freq = {}".format((k*scaling_ind+start_bin)*sr/n_fft))
            bins2freq.append((k*scaling_ind+start_bin)*sr/n_fft)
            binslist.append((k*scaling_ind+start_bin))
            wsin[k,0,:] = np.sin(2*np.pi*(k*scaling_ind+start_bin)*s/n_fft)
            wcos[k,0,:] = np.cos(2*np.pi*(k*scaling_ind+start_bin)*s/n_fft)

    elif freq_scale == 'log':
        if verbose==True:
            print(f"sampling rate = {sr}. Please make sure the sampling rate is correct in order to"
                  f"get a valid freq range")
        start_bin = start_freq*n_fft/sr
        scaling_ind = np.log(end_freq/start_freq)/freq_bins

        for k in range(freq_bins): # Only half of the bins contain useful info
            # print("log freq = {}".format(np.exp(k*scaling_ind)*start_bin*sr/n_fft))
            bins2freq.append(np.exp(k*scaling_ind)*start_bin*sr/n_fft)
            binslist.append((np.exp(k*scaling_ind)*start_bin))
            wsin[k,0,:] = np.sin(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)
            wcos[k,0,:] = np.cos(2*np.pi*(np.exp(k*scaling_ind)*start_bin)*s/n_fft)

    elif freq_scale == 'no':
        for k in range(freq_bins): # Only half of the bins contain useful info
            bins2freq.append(k*sr/n_fft)
            binslist.append(k)
            wsin[k,0,:] = np.sin(2*np.pi*k*s/n_fft)
            wcos[k,0,:] = np.cos(2*np.pi*k*s/n_fft)
    else:
        print("Please select the correct frequency scale, 'linear' or 'log'")
    return wsin.astype(np.float32),wcos.astype(np.float32), bins2freq, binslist, window_mask.astype(np.float32)


# Tools for CQT

def create_cqt_kernels(Q, fs, fmin, n_bins=84, bins_per_octave=12, norm=1,
                       window='hann', fmax=None, topbin_check=True):
    """
    Automatically create CQT kernels in time domain
    """

    fftLen = 2**nextpow2(np.ceil(Q * fs / fmin))
    # minWin = 2**nextpow2(np.ceil(Q * fs / fmax))

    if (fmax != None) and  (n_bins == None):
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    elif (fmax == None) and  (n_bins != None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    else:
        warnings.warn('If fmax is given, n_bins will be ignored',SyntaxWarning)
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    if np.max(freqs) > fs/2 and topbin_check==True:
        raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, \
                          please reduce the n_bins'.format(np.max(freqs)))

    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
    specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)

    lengths = np.ceil(Q * fs / freqs)
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = np.ceil(Q * fs / freq)

        # Centering the kernels
        if l%2==1: # pad more zeros on RHS
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))-1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))

        sig = get_window_dispatch(window,int(l), fftbins=True)*np.exp(np.r_[-l//2:l//2]*1j*2*np.pi*freq/fs)/l

        if norm: # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start:start + int(l)] = sig/np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start:start + int(l)] = sig
        # specKernel[k, :] = fft(tempKernel[k])

    # return specKernel[:,:fftLen//2+1], fftLen, torch.tensor(lenghts).float()
    return tempKernel, fftLen, torch.tensor(lengths).float(), freqs


def get_window_dispatch(window, N, fftbins=True):
    if isinstance(window, str):
        return get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == 'gaussian':
            assert window[1] >= 0
            sigma = np.floor(- N / 2 / np.sqrt(- 2 * np.log(10**(- window[1] / 20))))
            return get_window(('gaussian', sigma), N, fftbins=fftbins)
        else:
            Warning("Tuple windows may have undesired behaviour regarding Q factor")
    elif isinstance(window, float):
        Warning("You are using Kaiser window with beta factor " + str(window) + ". Correct behaviour not checked.")
    else:
        raise Exception("The function get_window from scipy only supports strings, tuples and floats.")



def get_cqt_complex(x, cqt_kernels_real, cqt_kernels_imag, hop_length, padding):
    """Multiplying the STFT result with the cqt_kernel, check out the 1992 CQT paper [1]
    for how to multiple the STFT result with the CQT kernel
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of
    a constant Q transform.” (1992)."""

    # STFT, converting the audio input from time domain to frequency domain
    try:
        x = padding(x) # When center == True, we need padding at the beginning and ending
    except:
        warnings.warn(f"\ninput size = {x.shape}\tkernel size = {cqt_kernels_real.shape[-1]}\n"
                      "padding with reflection mode might not be the best choice, try using constant padding",
                      UserWarning)        
        x = torch.nn.functional.pad(x, (cqt_kernels_real.shape[-1]//2, cqt_kernels_real.shape[-1]//2))
    CQT_real = conv1d(x, cqt_kernels_real, stride=hop_length)
    CQT_imag = -conv1d(x, cqt_kernels_imag, stride=hop_length)

    return torch.stack((CQT_real, CQT_imag),-1)

def get_cqt_complex2(x, cqt_kernels_real, cqt_kernels_imag, hop_length, padding, wcos=None, wsin=None):
    """Multiplying the STFT result with the cqt_kernel, check out the 1992 CQT paper [1]
    for how to multiple the STFT result with the CQT kernel
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of
    a constant Q transform.” (1992)."""

    # STFT, converting the audio input from time domain to frequency domain
    try:
        x = padding(x) # When center == True, we need padding at the beginning and ending
    except:
        warnings.warn(f"\ninput size = {x.shape}\tkernel size = {cqt_kernels_real.shape[-1]}\n"
                      "padding with reflection mode might not be the best choice, try using constant padding",
                      UserWarning)        
        x = torch.nn.functional.pad(x, (cqt_kernels_real.shape[-1]//2, cqt_kernels_real.shape[-1]//2))
        
   
    
    if wcos==None or wsin==None:
        CQT_real = conv1d(x, cqt_kernels_real, stride=hop_length)
        CQT_imag = -conv1d(x, cqt_kernels_imag, stride=hop_length)
        
    else:    
        fourier_real = conv1d(x, wcos, stride=hop_length)
        fourier_imag = conv1d(x, wsin, stride=hop_length)
        # Multiplying input with the CQT kernel in freq domain
        CQT_real, CQT_imag = complex_mul((cqt_kernels_real, cqt_kernels_imag),
                                         (fourier_real, fourier_imag)) 
        
    return torch.stack((CQT_real, CQT_imag),-1)




def create_lowpass_filter(band_center=0.5, kernelLength=256, transitionBandwidth=0.03):
    """
    Calculate the highest frequency we need to preserve and the lowest frequency we allow
    to pass through.
    Note that frequency is on a scale from 0 to 1 where 0 is 0 and 1 is Nyquist frequency of
    the signal BEFORE downsampling.
    """

    # transitionBandwidth = 0.03
    passbandMax = band_center / (1 + transitionBandwidth)
    stopbandMin = band_center * (1 + transitionBandwidth)

    # Unlike the filter tool we used online yesterday, this tool does
    # not allow us to specify how closely the filter matches our
    # specifications. Instead, we specify the length of the kernel.
    # The longer the kernel is, the more precisely it will match.
    # kernelLength = 256

    # We specify a list of key frequencies for which we will require
    # that the filter match a specific output gain.
    # From [0.0 to passbandMax] is the frequency range we want to keep
    # untouched and [stopbandMin, 1.0] is the range we want to remove
    keyFrequencies = [0.0, passbandMax, stopbandMin, 1.0]

    # We specify a list of output gains to correspond to the key
    # frequencies listed above.
    # The first two gains are 1.0 because they correspond to the first
    # two key frequencies. the second two are 0.0 because they
    # correspond to the stopband frequencies
    gainAtKeyFrequencies = [1.0, 1.0, 0.0, 0.0]

    # This command produces the filter kernel coefficients
    filterKernel = signal.firwin2(kernelLength, keyFrequencies, gainAtKeyFrequencies)

    return filterKernel.astype(np.float32)

def get_early_downsample_params(sr, hop_length, fmax_t, Q, n_octaves, verbose):
    """Used in CQT2010 and CQT2010v2"""
    
    window_bandwidth = 1.5 # for hann window
    filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)
    sr, hop_length, downsample_factor = early_downsample(sr,
                                                              hop_length,
                                                              n_octaves,
                                                              sr//2,
                                                              filter_cutoff)
    if downsample_factor != 1:
        if verbose==True:
            print("Can do early downsample, factor = ", downsample_factor)
        earlydownsample=True
        # print("new sr = ", sr)
        # print("new hop_length = ", hop_length)
        early_downsample_filter = create_lowpass_filter(band_center=1/downsample_factor,
                                                        kernelLength=256,
                                                        transitionBandwidth=0.03)
        early_downsample_filter = torch.tensor(early_downsample_filter)[None, None, :]

    else:
        if verbose==True:
            print("No early downsampling is required, downsample_factor = ", downsample_factor)
        early_downsample_filter = None
        earlydownsample=False

    return sr, hop_length, downsample_factor, early_downsample_filter, earlydownsample

def early_downsample(sr, hop_length, n_octaves,
                     nyquist, filter_cutoff):
    '''Return new sampling rate and hop length after early dowansampling'''
    downsample_count = early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves)
    # print("downsample_count = ", downsample_count)
    downsample_factor = 2**(downsample_count)

    hop_length //= downsample_factor # Getting new hop_length
    new_sr = sr / float(downsample_factor) # Getting new sampling rate
    sr = new_sr

    return sr, hop_length, downsample_factor


# The following two downsampling count functions are obtained from librosa CQT
# They are used to determine the number of pre resamplings if the starting and ending frequency
# are both in low frequency regions.
def early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    '''Compute the number of early downsampling operations'''

    downsample_count1 = max(0, int(np.ceil(np.log2(0.85 * nyquist /
                                                   filter_cutoff)) - 1) - 1)
    # print("downsample_count1 = ", downsample_count1)
    num_twos = nextpow2(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)
    # print("downsample_count2 = ",downsample_count2)

    return min(downsample_count1, downsample_count2)

def early_downsample(sr, hop_length, n_octaves,
                       nyquist, filter_cutoff):
    '''Return new sampling rate and hop length after early dowansampling'''
    downsample_count = early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves)
    # print("downsample_count = ", downsample_count)
    downsample_factor = 2**(downsample_count)

    hop_length //= downsample_factor  # Getting new hop_length
    new_sr = sr / float(downsample_factor)  # Getting new sampling rate

    sr = new_sr

    return sr, hop_length, downsample_factor