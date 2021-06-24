"""
Module containing all the spectrogram classes
"""

# 0.2.0

import torch
import torch.nn as nn
from torch.nn.functional import conv1d, conv2d, fold
import scipy # used only in CFP

import numpy as np
from time import time

# from nnAudio.librosa_functions import * # For debug purpose
# from nnAudio.utils import * 

from .librosa_functions import * 
from .utils import * 

sz_float = 4    # size of a float
epsilon = 10e-8 # fudge factor for normalization

### --------------------------- Spectrogram Classes ---------------------------###
class STFT(torch.nn.Module):
    """This function is to calculate the short-time Fourier transform (STFT) of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred automatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.

    Parameters
    ----------
    n_fft : int
        Size of Fourier transform. Default value is 2048.

    win_length : int
        the size of window frame and STFT filter.
        Default: None (treated as equal to n_fft)

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins.

    hop_length : int
        The hop (or stride) size. Default value is ``None`` which is equivalent to ``n_fft//4``.

    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.

    freq_scale : 'linear', 'log', or 'no'
        Determine the spacing between each frequency bin. When `linear` or `log` is used,
        the bin spacing can be controlled by ``fmin`` and ``fmax``. If 'no' is used, the bin will
        start at 0Hz and end at Nyquist frequency with linear spacing.

    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``, the time
        index is the beginning of the STFT kernel, if ``True``, the time index is the center of
        the STFT kernel. Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.
       
    iSTFT : bool
        To activate the iSTFT module or not. By default, it is False to save GPU memory.
        Note: The iSTFT kernel is not trainable. If you want
        a trainable iSTFT, use the iSTFT module.

    fmin : int
        The starting frequency for the lowest frequency bin. If freq_scale is ``no``, this argument
        does nothing.

    fmax : int
        The ending frequency for the highest frequency bin. If freq_scale is ``no``, this argument
        does nothing.

    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    trainable : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT
        kernels will also be caluclated and the STFT kernels will be updated during model training.
        Default value is ``False``
        
    output_format : str
        Control the spectrogram output type, either ``Magnitude``, ``Complex``, or ``Phase``.
        The output_format can also be changed during the ``forward`` method.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints
    
    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.
        ``shape = (num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
        ``shape = (num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;        

    Examples
    --------
    >>> spec_layer = Spectrogram.STFT()
    >>> specs = spec_layer(x)
    """

    def __init__(self, n_fft=2048, win_length=None, freq_bins=None, hop_length=None, window='hann',
                freq_scale='no', center=True, pad_mode='reflect', iSTFT=False,
                fmin=50, fmax=6000, sr=22050, trainable=False,
                output_format="Complex", verbose=True):

        super().__init__()

        # Trying to make the default setting same as librosa
        if win_length==None: win_length = n_fft
        if hop_length==None: hop_length = int(win_length // 4)

        self.output_format = output_format
        self.trainable = trainable
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.freq_bins = freq_bins
        self.trainable = trainable
        self.pad_amount = self.n_fft // 2
        self.window = window
        self.win_length = win_length
        self.iSTFT = iSTFT
        self.trainable = trainable
        start = time()



        # Create filter windows for stft
        kernel_sin, kernel_cos, self.bins2freq, self.bin_list, window_mask = create_fourier_kernels(n_fft,
                                                                           win_length=win_length,
                                                                           freq_bins=freq_bins,
                                                                           window=window,
                                                                           freq_scale=freq_scale,
                                                                           fmin=fmin,
                                                                           fmax=fmax,
                                                                           sr=sr,
                                                                           verbose=verbose)


        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float)
        
        # In this way, the inverse kernel and the forward kernel do not share the same memory...
        kernel_sin_inv = torch.cat((kernel_sin, -kernel_sin[1:-1].flip(0)), 0)
        kernel_cos_inv = torch.cat((kernel_cos, kernel_cos[1:-1].flip(0)), 0)
        
         
        
        if iSTFT:
            self.register_buffer('kernel_sin_inv', kernel_sin_inv.unsqueeze(-1))
            self.register_buffer('kernel_cos_inv', kernel_cos_inv.unsqueeze(-1))

        # Making all these variables nn.Parameter, so that the model can be used with nn.Parallel
#         self.kernel_sin = torch.nn.Parameter(self.kernel_sin, requires_grad=self.trainable)
#         self.kernel_cos = torch.nn.Parameter(self.kernel_cos, requires_grad=self.trainable)

        # Applying window functions to the Fourier kernels
        if window:
            window_mask = torch.tensor(window_mask)
            wsin = kernel_sin * window_mask
            wcos = kernel_cos * window_mask
        else:
            wsin = kernel_sin
            wcos = kernel_cos
        
        if self.trainable==False:
            self.register_buffer('wsin', wsin)
            self.register_buffer('wcos', wcos)            
        
        if self.trainable==True:
            wsin = torch.nn.Parameter(wsin, requires_grad=self.trainable)
            wcos = torch.nn.Parameter(wcos, requires_grad=self.trainable)  
            self.register_parameter('wsin', wsin)
            self.register_parameter('wcos', wcos)         
        
        # Prepare the shape of window mask so that it can be used later in inverse
        self.register_buffer('window_mask', window_mask.unsqueeze(0).unsqueeze(-1))
        
            

        if verbose==True:
            print("STFT kernels created, time used = {:.4f} seconds".format(time()-start))
        else:
            pass

    def forward(self, x, output_format=None):
        """
        Convert a batch of waveforms to spectrograms.
        
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        
        output_format : str
            Control the type of spectrogram to be return. Can be either ``Magnitude`` or ``Complex`` or ``Phase``.
            Default value is ``Complex``.  
            
        """
        output_format = output_format or self.output_format
        self.num_samples = x.shape[-1]
        
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.pad_amount, 0)

            elif self.pad_mode == 'reflect':
                if self.num_samples < self.pad_amount:
                    raise AssertionError("Signal length shorter than reflect padding length (n_fft // 2).")
                padding = nn.ReflectionPad1d(self.pad_amount)

            x = padding(x)
        spec_imag = conv1d(x, self.wsin, stride=self.stride)
        spec_real = conv1d(x, self.wcos, stride=self.stride)  # Doing STFT by using conv1d

        # remove redundant parts
        spec_real = spec_real[:, :self.freq_bins, :]
        spec_imag = spec_imag[:, :self.freq_bins, :]

        if output_format=='Magnitude':
            spec = spec_real.pow(2) + spec_imag.pow(2)
            if self.trainable==True:
                return torch.sqrt(spec+1e-8)  # prevent Nan gradient when sqrt(0) due to output=0
            else:
                return torch.sqrt(spec)

        elif output_format=='Complex':
            return torch.stack((spec_real,-spec_imag), -1)  # Remember the minus sign for imaginary part

        elif output_format=='Phase':
            return torch.atan2(-spec_imag+0.0,spec_real)  # +0.0 removes -0.0 elements, which leads to error in calculating phase

    def inverse(self, X, onesided=True, length=None, refresh_win=True):
        """
        This function is same as the :func:`~nnAudio.Spectrogram.iSTFT` class, 
        which is to convert spectrograms back to waveforms. 
        It only works for the complex value spectrograms. If you have the magnitude spectrograms,
        please use :func:`~nnAudio.Spectrogram.Griffin_Lim`. 
        
        Parameters
        ----------
        onesided : bool
            If your spectrograms only have ``n_fft//2+1`` frequency bins, please use ``onesided=True``,
            else use ``onesided=False``

        length : int
            To make sure the inverse STFT has the same output length of the original waveform, please
            set `length` as your intended waveform length. By default, ``length=None``,
            which will remove ``n_fft//2`` samples from the start and the end of the output.
            
        refresh_win : bool
            Recalculating the window sum square. If you have an input with fixed number of timesteps,
            you can increase the speed by setting ``refresh_win=False``. Else please keep ``refresh_win=True``
           
           
        """
        if (hasattr(self, 'kernel_sin_inv') != True) or (hasattr(self, 'kernel_cos_inv') != True):
            raise NameError("Please activate the iSTFT module by setting `iSTFT=True` if you want to use `inverse`")      
        
        assert X.dim()==4 , "Inverse iSTFT only works for complex number," \
                            "make sure our tensor is in the shape of (batch, freq_bins, timesteps, 2)."\
                            "\nIf you have a magnitude spectrogram, please consider using Griffin-Lim."
        if onesided:
            X = extend_fbins(X) # extend freq

    
        X_real, X_imag = X[:, :, :, 0], X[:, :, :, 1]

        # broadcast dimensions to support 2D convolution
        X_real_bc = X_real.unsqueeze(1)
        X_imag_bc = X_imag.unsqueeze(1)
        a1 = conv2d(X_real_bc, self.kernel_cos_inv, stride=(1,1))
        b2 = conv2d(X_imag_bc, self.kernel_sin_inv, stride=(1,1))
       
        # compute real and imag part. signal lies in the real part
        real = a1 - b2
        real = real.squeeze(-2)*self.window_mask

        # Normalize the amplitude with n_fft
        real /= (self.n_fft)

        # Overlap and Add algorithm to connect all the frames
        real = overlap_add(real, self.stride)
    
        # Prepare the window sumsqure for division
        # Only need to create this window once to save time
        # Unless the input spectrograms have different time steps
        if hasattr(self, 'w_sum')==False or refresh_win==True:
            self.w_sum = torch_window_sumsquare(self.window_mask.flatten(), X.shape[2], self.stride, self.n_fft).flatten()
            self.nonzero_indices = (self.w_sum>1e-10)    
        else:
            pass
        real[:, self.nonzero_indices] = real[:,self.nonzero_indices].div(self.w_sum[self.nonzero_indices])
        # Remove padding
        if length is None:       
            if self.center:
                real = real[:, self.pad_amount:-self.pad_amount]

        else:
            if self.center:
                real = real[:, self.pad_amount:self.pad_amount + length]    
            else:
                real = real[:, :length] 
            
        return real
    
    def extra_repr(self) -> str:
        return 'n_fft={}, Fourier Kernel size={}, iSTFT={}, trainable={}'.format(
            self.n_fft, (*self.wsin.shape,), self.iSTFT, self.trainable
        )    


class MelSpectrogram(torch.nn.Module):
    """This function is to calculate the Melspectrogram of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred automatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio.
        It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    n_fft : int
        The window size for the STFT. Default value is 2048

    win_length : int
        the size of window frame and STFT filter.
        Default: None (treated as equal to n_fft)

    n_mels : int
        The number of Mel filter banks. The filter banks maps the n_fft to mel bins.
        Default value is 128.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.

    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``,
        the time index is the beginning of the STFT kernel, if ``True``, the time index is the
        center of the STFT kernel. Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    htk : bool
        When ``False`` is used, the Mel scale is quasi-logarithmic. When ``True`` is used, the
        Mel scale is logarithmic. The default value is ``False``.

    fmin : int
        The starting frequency for the lowest Mel filter bank.

    fmax : int
        The ending frequency for the highest Mel filter bank.

    norm :
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization, AKA 'slaney' default in librosa).
        Otherwise, leave all the triangles aiming for
        a peak value of 1.0

    trainable_mel : bool
        Determine if the Mel filter banks are trainable or not. If ``True``, the gradients for Mel
        filter banks will also be calculated and the Mel filter banks will be updated during model
        training. Default value is ``False``.

    trainable_STFT : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT
        kernels will also be caluclated and the STFT kernels will be updated during model training.
        Default value is ``False``.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.  shape = ``(num_samples, freq_bins,time_steps)``.

    Examples
    --------
    >>> spec_layer = Spectrogram.MelSpectrogram()
    >>> specs = spec_layer(x)
    """

    def __init__(self, sr=22050, n_fft=2048, win_length=None, n_mels=128, hop_length=512, 
                window='hann', center=True, pad_mode='reflect', power=2.0, htk=False, 
                fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False, 
                verbose=True, **kwargs):

        super().__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.power = power
        self.trainable_mel = trainable_mel
        self.trainable_STFT = trainable_STFT

        # Preparing for the stft layer. No need for center
        self.stft = STFT(n_fft=n_fft, win_length=win_length, freq_bins=None,
                hop_length=hop_length, window=window, freq_scale='no',
                center=center, pad_mode=pad_mode, sr=sr, trainable=trainable_STFT,
                output_format="Magnitude", verbose=verbose, **kwargs)
        
        
        # Create filter windows for stft
        start = time()

        # Creating kernel for mel spectrogram
        start = time()
        mel_basis = mel(sr, n_fft, n_mels, fmin, fmax, htk=htk, norm=norm)
        mel_basis = torch.tensor(mel_basis)

        if verbose==True:
            print("STFT filter created, time used = {:.4f} seconds".format(time()-start))
            print("Mel filter created, time used = {:.4f} seconds".format(time()-start))
        else:
            pass

        if trainable_mel:
        # Making everything nn.Parameter, so that this model can support nn.DataParallel
            mel_basis = torch.nn.Parameter(mel_basis, requires_grad=trainable_mel)
            self.register_parameter('mel_basis', mel_basis)
        else:
            self.register_buffer('mel_basis', mel_basis)

        # if trainable_mel==True:
        #     self.mel_basis = torch.nn.Parameter(self.mel_basis)
        # if trainable_STFT==True:
        #     self.wsin = torch.nn.Parameter(self.wsin)
        #     self.wcos = torch.nn.Parameter(self.wcos)

    def forward(self, x):
        """
        Convert a batch of waveforms to Mel spectrograms.
        
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """        
        x = broadcast_dim(x)
        
        spec = self.stft(x, output_format='Magnitude')**self.power

        melspec = torch.matmul(self.mel_basis, spec)
        return melspec
    
    def extra_repr(self) -> str:
        return 'Mel filter banks size = {}, trainable_mel={}'.format(
            (*self.mel_basis.shape,), self.trainable_mel, self.trainable_STFT
        )        


class MFCC(torch.nn.Module):
    """This function is to calculate the Mel-frequency cepstral coefficients (MFCCs) of the input signal.
    This algorithm first extracts Mel spectrograms from the audio clips,
    then the discrete cosine transform is calcuated to obtain the final MFCCs.
    Therefore, the Mel spectrogram part can be made trainable using 
    ``trainable_mel`` and ``trainable_STFT``.
    It only support type-II DCT at the moment. Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio.  It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    n_mfcc : int
        The number of Mel-frequency cepstral coefficients

    norm : string
        The default value is 'ortho'. Normalization for DCT basis

    **kwargs
        Other arguments for Melspectrogram such as n_fft, n_mels, hop_length, and window

    Returns
    -------
    MFCCs : torch.tensor
        It returns a tensor of MFCCs.  shape = ``(num_samples, n_mfcc, time_steps)``.

    Examples
    --------
    >>> spec_layer = Spectrogram.MFCC()
    >>> mfcc = spec_layer(x)
    """

    def __init__(self, sr=22050, n_mfcc=20, norm='ortho', verbose=True, ref=1.0, amin=1e-10, top_db=80.0, **kwargs):
        super().__init__()
        self.melspec_layer = MelSpectrogram(sr=sr, verbose=verbose, **kwargs)
        self.m_mfcc = n_mfcc
        
        # attributes that will be used for _power_to_db
        if amin <= 0:
            raise ParameterError('amin must be strictly positive')
        amin = torch.tensor([amin])
        ref = torch.abs(torch.tensor([ref]))
        self.register_buffer('amin', amin)
        self.register_buffer('ref', ref)    
        self.top_db = top_db
        self.n_mfcc = n_mfcc

    def _power_to_db(self, S):
        '''
        Refer to https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#power_to_db
        for the original implmentation.
        '''

        log_spec = 10.0 * torch.log10(torch.max(S, self.amin))
        log_spec -= 10.0 * torch.log10(torch.max(self.amin, self.ref))
        if self.top_db is not None:
            if self.top_db < 0:
                raise ParameterError('top_db must be non-negative')

            # make the dim same as log_spec so that it can be broadcasted
            batch_wise_max = log_spec.flatten(1).max(1)[0].unsqueeze(1).unsqueeze(1)
            log_spec = torch.max(log_spec, batch_wise_max - self.top_db)

        return log_spec     
    
    def _dct(self, x, norm=None):
        '''
        Refer to https://github.com/zh217/torch-dct for the original implmentation.
        '''
        x = x.permute(0,2,1) # make freq the last axis, since dct applies to the frequency axis
        x_shape = x.shape
        N = x_shape[-1]

        v = torch.cat([x[:, :, ::2], x[:, :, 1::2].flip([2])], dim=2)
        Vc = torch.rfft(v, 1, onesided=False)

        # TODO: Can make the W_r and W_i trainable here
        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, :, 0] * W_r - Vc[:, :, :, 1] * W_i

        if norm == 'ortho':
            V[:, :, 0] /= np.sqrt(N) * 2
            V[:, :, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V

        return V.permute(0,2,1)  # swapping back the time axis and freq axis    

    def forward(self, x):
        """
        Convert a batch of waveforms to MFCC.
        
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``      
            It will be automatically broadcast to the right shape
        """           
        
        x = self.melspec_layer(x)
        x = self._power_to_db(x)
        x = self._dct(x, norm='ortho')[:,:self.m_mfcc,:]
        return x

    def extra_repr(self) -> str:
        return 'n_mfcc = {}'.format(
            (self.n_mfcc)        
        )


class Gammatonegram(torch.nn.Module):
    """
    This function is to calculate the Gammatonegram of the input signal. Input signal should be in either of the following shapes. 1. ``(len_audio)``, 2. ``(num_audio, len_audio)``, 3. ``(num_audio, 1, len_audio)``. The correct shape will be inferred autommatically if the input follows these 3 shapes. This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.
    
    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``. Setting the correct sampling rate is very important for calculating the correct frequency.
    n_fft : int
        The window size for the STFT. Default value is 2048
    n_mels : int
        The number of Gammatonegram filter banks. The filter banks maps the n_fft to Gammatone bins. Default value is 64

    hop_length : int
        The hop (or stride) size. Default value is 512.
    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to scipy documentation for possible windowing functions. The default value is 'hann'
    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``, the time index is the beginning of the STFT kernel, if ``True``, the time index is the center of the STFT kernel. Default value if ``True``.
    pad_mode : str
        The padding method. Default value is 'reflect'.
    htk : bool
        When ``False`` is used, the Mel scale is quasi-logarithmic. When ``True`` is used, the Mel scale is logarithmic. The default value is ``False``

    fmin : int
        The starting frequency for the lowest Gammatone filter bank
    fmax : int
        The ending frequency for the highest Gammatone filter bank
    trainable_mel : bool
        Determine if the Gammatone filter banks are trainable or not. If ``True``, the gradients for Mel filter banks will also be caluclated and the Mel filter banks will be updated during model training. Default value is ``False``
    trainable_STFT : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT kernels will also be caluclated and the STFT kernels will be updated during model training. Default value is ``False``

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.  shape = ``(num_samples, freq_bins,time_steps)``.
        
    Examples
    --------
    >>> spec_layer = Spectrogram.Gammatonegram()
    >>> specs = spec_layer(x)
    """

    def __init__(self, sr=44100, n_fft=2048, n_bins=64, hop_length=512, window='hann', center=True, pad_mode='reflect',
                 power=2.0, htk=False, fmin=20.0, fmax=None, norm=1, trainable_bins=False, trainable_STFT=False,
                 verbose=True):
        super(Gammatonegram, self).__init__()
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.power = power

        # Create filter windows for stft
        start = time()
        wsin, wcos, self.bins2freq, _, _ = create_fourier_kernels(n_fft, freq_bins=None, window=window, freq_scale='no',
                                                               sr=sr)
        
        wsin = torch.tensor(wsin, dtype=torch.float)
        wcos = torch.tensor(wcos, dtype=torch.float)
        
        if trainable_STFT:
            wsin = torch.nn.Parameter(wsin, requires_grad=trainable_STFT)
            wcos = torch.nn.Parameter(wcos, requires_grad=trainable_STFT)
            self.register_parameter('wsin', wsin)
            self.register_parameter('wcos', wcos)               
        else:
            self.register_buffer('wsin', wsin)
            self.register_buffer('wcos', wcos)   

        # Creating kenral for Gammatone spectrogram
        start = time()
        gammatone_basis = gammatone(sr, n_fft, n_bins, fmin, fmax)
        gammatone_basis = torch.tensor(gammatone_basis)

        if verbose == True:
            print("STFT filter created, time used = {:.4f} seconds".format(time() - start))
            print("Gammatone filter created, time used = {:.4f} seconds".format(time() - start))
        else:
            pass
        # Making everything nn.Prarmeter, so that this model can support nn.DataParallel
        
        if trainable_bins:   
            gammatone_basis = torch.nn.Parameter(gammatone_basis, requires_grad=trainable_bins)
            self.register_parameter('gammatone_basis', gammatone_basis)   
        else:
            self.register_buffer('gammatone_basis', gammatone_basis)

        # if trainable_mel==True:
        #     self.mel_basis = torch.nn.Parameter(self.mel_basis)
        # if trainable_STFT==True:
        #     self.wsin = torch.nn.Parameter(self.wsin)
        #     self.wcos = torch.nn.Parameter(self.wcos)

    def forward(self, x):
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.n_fft // 2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.n_fft // 2)

            x = padding(x)

        spec = torch.sqrt(conv1d(x, self.wsin, stride=self.stride).pow(2) \
                          + conv1d(x, self.wcos, stride=self.stride).pow(2)) ** self.power  # Doing STFT by using conv1d

        gammatonespec = torch.matmul(self.gammatone_basis, spec)
        return gammatonespec


class CQT1992(torch.nn.Module):
    """
    This alogrithm uses the method proposed in [1], which would run extremely slow if low frequencies (below 220Hz)
    are included in the frequency bins.
    Please refer to :func:`~nnAudio.Spectrogram.CQT1992v2` for a more
    computational and memory efficient version.
    [1] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).    
    
    This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.



    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.
        If ``fmax`` is not ``None``, then the argument ``n_bins`` will be ignored and ``n_bins``
        will be calculated automatically. Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.
        
    trainable_STFT : bool
        Determine if the time to frequency domain transformation kernel for the input audio is trainable or not.
        Default is ``False``
        
    trainable_CQT : bool
        Determine if the frequency domain CQT kernel is trainable or not.
        Default is ``False``        

    norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : str
        The windowing function for CQT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.

    center : bool
        Putting the CQT keneral at the center of the time-step or not. If ``False``, the time index is
        the beginning of the CQT kernel, if ``True``, the time index is the center of the CQT kernel.
        Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``.

     output_format : str
        Determine the return type.
        ``Magnitude`` will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``;
        ``Complex`` will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``;
        ``Phase`` will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    Returns
    -------
    spectrogram : torch.tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT1992v2()
    >>> specs = spec_layer(x)
    """
    
    def __init__(self, sr=22050, hop_length=512, fmin=220, fmax=None, n_bins=84,
                 trainable_STFT=False, trainable_CQT=False, bins_per_octave=12, filter_scale=1,
                 output_format='Magnitude', norm=1, window='hann', center=True, pad_mode='reflect'):

        super().__init__()

        # norm arg is not functioning
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.norm = norm
        self.output_format = output_format

        # creating kernels for CQT
        Q = float(filter_scale)/(2**(1/bins_per_octave)-1)

        print("Creating CQT kernels ...", end='\r')
        start = time()
        cqt_kernels, self.kernel_width, lenghts, freqs = create_cqt_kernels(Q,
                                                                sr,
                                                                fmin,
                                                                n_bins,
                                                                bins_per_octave,
                                                                norm,
                                                                window,
                                                                fmax)
        
        self.register_buffer('lenghts', lenghts)
        self.frequencies = freqs
        
        cqt_kernels = fft(cqt_kernels)[:,:self.kernel_width//2+1]
        print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))

        # creating kernels for stft
        # self.cqt_kernels_real*=lenghts.unsqueeze(1)/self.kernel_width # Trying to normalize as librosa
        # self.cqt_kernels_imag*=lenghts.unsqueeze(1)/self.kernel_width

        print("Creating STFT kernels ...", end='\r')
        start = time()
        kernel_sin, kernel_cos, self.bins2freq, _, window = create_fourier_kernels(self.kernel_width,
                                                               window='ones',
                                                               freq_scale='no')

        # Converting kernels from numpy arrays to torch tensors
        wsin = torch.tensor(kernel_sin * window)
        wcos = torch.tensor(kernel_cos * window) 
        
        cqt_kernels_real = torch.tensor(cqt_kernels.real.astype(np.float32))
        cqt_kernels_imag = torch.tensor(cqt_kernels.imag.astype(np.float32))
        
        if trainable_STFT:
            wsin = torch.nn.Parameter(wsin, requires_grad=trainable_STFT)
            wcos = torch.nn.Parameter(wcos, requires_grad=trainable_STFT)
            self.register_parameter('wsin', wsin)
            self.register_parameter('wcos', wcos)                   
        else:
            self.register_buffer('wsin', wsin)
            self.register_buffer('wcos', wcos)
            
        if trainable_CQT:
            cqt_kernels_real = torch.nn.Parameter(cqt_kernels_real, requires_grad=trainable_CQT)
            cqt_kernels_imag = torch.nn.Parameter(cqt_kernels_imag, requires_grad=trainable_CQT)
            self.register_parameter('cqt_kernels_real', cqt_kernels_real)
            self.register_parameter('cqt_kernels_imag', cqt_kernels_imag)
        else:
            self.register_buffer('cqt_kernels_real', cqt_kernels_real)
            self.register_buffer('cqt_kernels_imag', cqt_kernels_imag)                  

        print("STFT kernels created, time used = {:.4f} seconds".format(time()-start))

    def forward(self, x, output_format=None, normalization_type='librosa'):
        """
        Convert a batch of waveforms to CQT spectrograms.
        
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """         
        output_format = output_format or self.output_format
        
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.kernel_width//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.kernel_width//2)

            x = padding(x)           

        # STFT
        fourier_real = conv1d(x, self.wcos, stride=self.hop_length)
        fourier_imag = conv1d(x, self.wsin, stride=self.hop_length)

        # CQT
        CQT_real, CQT_imag = complex_mul((self.cqt_kernels_real, self.cqt_kernels_imag),
                                         (fourier_real, fourier_imag))
        
        CQT = torch.stack((CQT_real,-CQT_imag),-1)

        if normalization_type == 'librosa':
            CQT *= torch.sqrt(self.lenghts.view(-1,1,1))/self.kernel_width
        elif normalization_type == 'convolutional':
            pass
        elif normalization_type == 'wrap':
            CQT *= 2/self.kernel_width
        else:
            raise ValueError("The normalization_type %r is not part of our current options." % normalization_type)        
        
        
#         if self.norm:
#             CQT = CQT/self.kernel_width*torch.sqrt(self.lenghts.view(-1,1,1))
#         else:
#             CQT = CQT*torch.sqrt(self.lenghts.view(-1,1,1))

        if output_format=='Magnitude':
            # Getting CQT Amplitude
            return torch.sqrt(CQT.pow(2).sum(-1))

        elif output_format=='Complex':
            return CQT
        
        elif output_format=='Phase':
            phase_real = torch.cos(torch.atan2(CQT_imag,CQT_real))
            phase_imag = torch.sin(torch.atan2(CQT_imag,CQT_real))
            return torch.stack((phase_real,phase_imag), -1)     
        
    def extra_repr(self) -> str:
        return 'STFT kernel size = {}, CQT kernel size = {}'.format(
            (*self.wcos.shape,), (*self.cqt_kernels_real.shape,)        
        )             


class CQT2010(torch.nn.Module):
    """
    This algorithm is using the resampling method proposed in [1].
    Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency
    spectrum, we make a small CQT kernel covering only the top octave.
    Then we keep downsampling the input audio by a factor of 2 to convoluting it with the
    small CQT kernel. Everytime the input audio is downsampled, the CQT relative to the downsampled
    input is equavalent to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the code
    from the 1992 alogrithm [2]
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).
    early downsampling factor is to downsample the input audio to reduce the CQT kernel size.
    The result with and without early downsampling are more or less the same except in the very low
    frequency region where freq < 40Hz.
    """

    def __init__(self, sr=22050, hop_length=512, fmin=32.70, fmax=None, n_bins=84, bins_per_octave=12,
                 norm=True, basis_norm=1, window='hann', pad_mode='reflect', trainable_STFT=False, filter_scale=1,
                 trainable_CQT=False, output_format='Magnitude', earlydownsample=True, verbose=True):

        super().__init__()

        self.norm = norm  # Now norm is used to normalize the final CQT result by dividing n_fft
        # basis_norm is for normalizing basis
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins
        self.output_format = output_format
        self.earlydownsample = earlydownsample  # TODO: activate early downsampling later if possible

        # This will be used to calculate filter_cutoff and creating CQT kernels
        Q = float(filter_scale)/(2**(1/bins_per_octave)-1)

        # Creating lowpass filter and make it a torch tensor
        if verbose==True:
            print("Creating low pass filter ...", end='\r')
        start = time()
        lowpass_filter = torch.tensor(create_lowpass_filter(
                                                            band_center = 0.5,
                                                            kernelLength=256,
                                                            transitionBandwidth=0.001
                                                            )
                                     )

        # Broadcast the tensor to the shape that fits conv1d
        self.register_buffer('lowpass_filter', lowpass_filter[None,None,:])

        if verbose==True:
            print("Low pass filter created, time used = {:.4f} seconds".format(time()-start))

        # Calculate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        # print("n_octaves = ", self.n_octaves)

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = fmin*2**(self.n_octaves-1)
        remainder = n_bins % bins_per_octave
        # print("remainder = ", remainder)

        if remainder==0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t*2**((bins_per_octave-1)/bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t*2**((remainder-1)/bins_per_octave)

        self.fmin_t = fmax_t/2**(1-1/bins_per_octave) # Adjusting the top minium bins
        if fmax_t > sr/2:
            raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, \
                              please reduce the n_bins'.format(fmax_t))

        if self.earlydownsample == True: # Do early downsampling if this argument is True
            if verbose==True:
                print("Creating early downsampling filter ...", end='\r')
            start = time()
            sr, self.hop_length, self.downsample_factor, early_downsample_filter, \
                self.earlydownsample = get_early_downsample_params(sr,
                                                                        hop_length,
                                                                        fmax_t,
                                                                        Q,
                                                                        self.n_octaves,
                                                                        verbose)
            
            self.register_buffer('early_downsample_filter', early_downsample_filter)
            if verbose==True:
                print("Early downsampling filter created, \
                            time used = {:.4f} seconds".format(time()-start))
        else:
            self.downsample_factor=1.

        # Preparing CQT kernels
        if verbose==True:
            print("Creating CQT kernels ...", end='\r')

        start = time()
        # print("Q = {}, fmin_t = {}, n_filters = {}".format(Q, self.fmin_t, n_filters))
        basis, self.n_fft, _, _ = create_cqt_kernels(Q,
                                                  sr,
                                                  self.fmin_t,
                                                  n_filters,
                                                  bins_per_octave,
                                                  norm=basis_norm,
                                                  topbin_check=False)

        # This is for the normalization in the end
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        self.frequencies = freqs
        
        lenghts = np.ceil(Q * sr / freqs)
        lenghts = torch.tensor(lenghts).float()
        self.register_buffer('lenghts', lenghts)


        self.basis=basis
        fft_basis = fft(basis)[:,:self.n_fft//2+1]  # Convert CQT kenral from time domain to freq domain

        # These cqt_kernel is already in the frequency domain
        cqt_kernels_real = torch.tensor(fft_basis.real.astype(np.float32))
        cqt_kernels_imag = torch.tensor(fft_basis.imag.astype(np.float32))

        if verbose==True:
            print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))

        # print("Getting cqt kernel done, n_fft = ",self.n_fft)
        # Preparing kernels for Short-Time Fourier Transform (STFT)
        # We set the frequency range in the CQT filter instead of here.

        if verbose==True:
            print("Creating STFT kernels ...", end='\r')

        start = time()
        kernel_sin, kernel_cos, self.bins2freq, _, window = create_fourier_kernels(self.n_fft, window='ones', freq_scale='no')
        wsin = kernel_sin * window
        wcos = kernel_cos * window
        
        wsin = torch.tensor(wsin)
        wcos = torch.tensor(wcos)
        
        if verbose==True:
            print("STFT kernels created, time used = {:.4f} seconds".format(time()-start))

        if trainable_STFT:
            wsin = torch.nn.Parameter(wsin, requires_grad=trainable_STFT)
            wcos = torch.nn.Parameter(wcos, requires_grad=trainable_STFT)
            self.register_parameter('wsin', wsin)
            self.register_parameter('wcos', wcos)                   
        else:
            self.register_buffer('wsin', wsin)
            self.register_buffer('wcos', wcos)
            
        if trainable_CQT:
            cqt_kernels_real = torch.nn.Parameter(cqt_kernels_real, requires_grad=trainable_CQT)
            cqt_kernels_imag = torch.nn.Parameter(cqt_kernels_imag, requires_grad=trainable_CQT)
            self.register_parameter('cqt_kernels_real', cqt_kernels_real)
            self.register_parameter('cqt_kernels_imag', cqt_kernels_imag)
        else:
            self.register_buffer('cqt_kernels_real', cqt_kernels_real)
            self.register_buffer('cqt_kernels_imag', cqt_kernels_imag)  
            
        # If center==True, the STFT window will be put in the middle, and paddings at the beginning
        # and ending are required.
        if self.pad_mode == 'constant':
            self.padding = nn.ConstantPad1d(self.n_fft//2, 0)
        elif self.pad_mode == 'reflect':
            self.padding = nn.ReflectionPad1d(self.n_fft//2)


    def forward(self,x, output_format=None, normalization_type='librosa'):
        """
        Convert a batch of waveforms to CQT spectrograms.
        
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """              
        output_format = output_format or self.output_format
        
        x = broadcast_dim(x)
        if self.earlydownsample==True:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor)
        hop = self.hop_length
        

        
        CQT = get_cqt_complex2(x, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding,
                               wcos=self.wcos, wsin=self.wsin)

        x_down = x  # Preparing a new variable for downsampling
        for i in range(self.n_octaves-1):
            hop = hop//2
            x_down = downsampling_by_2(x_down, self.lowpass_filter)
            
            CQT1 = get_cqt_complex2(x_down, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding,
                                    wcos=self.wcos, wsin=self.wsin)          
            CQT = torch.cat((CQT1, CQT),1)
             
        CQT = CQT[:,-self.n_bins:,:]  # Removing unwanted top bins

        if normalization_type == 'librosa':
            CQT *= torch.sqrt(self.lenghts.view(-1,1,1))/self.n_fft
        elif normalization_type == 'convolutional':
            pass
        elif normalization_type == 'wrap':
            CQT *= 2/self.n_fft
        else:
            raise ValueError("The normalization_type %r is not part of our current options." % normalization_type)                
        
        if output_format=='Magnitude':
            # Getting CQT Amplitude
            return torch.sqrt(CQT.pow(2).sum(-1))
            
        elif output_format=='Complex':
            return CQT
        
        elif output_format=='Phase':
            phase_real = torch.cos(torch.atan2(CQT[:,:,:,1],CQT[:,:,:,0]))
            phase_imag = torch.sin(torch.atan2(CQT[:,:,:,1],CQT[:,:,:,0]))
            return torch.stack((phase_real,phase_imag), -1)             

    def extra_repr(self) -> str:
        return 'STFT kernel size = {}, CQT kernel size = {}'.format(
            (*self.wcos.shape,), (*self.cqt_kernels_real.shape,)        
        )    
        

class CQT1992v2(torch.nn.Module):
    """This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.

    This alogrithm uses the method proposed in [1]. I slightly modify it so that it runs faster
    than the original 1992 algorithm, that is why I call it version 2.
    [1] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.
        If ``fmax`` is not ``None``, then the argument ``n_bins`` will be ignored and ``n_bins``
        will be calculated automatically. Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    filter_scale : float > 0
        Filter scale factor. Values of filter_scale smaller than 1 can be used to improve the time resolution at the
        cost of degrading the frequency resolution. Important to note is that setting for example filter_scale = 0.5 and
        bins_per_octave = 48 leads to exactly the same time-frequency resolution trade-off as setting filter_scale = 1
        and bins_per_octave = 24, but the former contains twice more frequency bins per octave. In this sense, values
        filter_scale < 1 can be seen to implement oversampling of the frequency axis, analogously to the use of zero
        padding when calculating the DFT.

    norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : string, float, or tuple
        The windowing function for CQT. If it is a string, It uses ``scipy.signal.get_window``. If it is a
        tuple, only the gaussian window wanrantees constant Q factor. Gaussian window should be given as a
        tuple ('gaussian', att) where att is the attenuation in the border given in dB.
        Please refer to scipy documentation for possible windowing functions. The default value is 'hann'.

    center : bool
        Putting the CQT keneral at the center of the time-step or not. If ``False``, the time index is
        the beginning of the CQT kernel, if ``True``, the time index is the center of the CQT kernel.
        Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``.

    output_format : str
        Determine the return type.
        ``Magnitude`` will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins,time_steps)``;
        ``Complex`` will return the STFT result in complex number, shape = ``(num_samples, freq_bins,time_steps, 2)``;
        ``Phase`` will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    Returns
    -------
    spectrogram : torch.tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT1992v2()
    >>> specs = spec_layer(x)
    """

    def __init__(self, sr=22050, hop_length=512, fmin=32.70, fmax=None, n_bins=84,
                 bins_per_octave=12, filter_scale=1, norm=1, window='hann', center=True, pad_mode='reflect',
                 trainable=False, output_format='Magnitude', verbose=True):

        super().__init__()

        self.trainable = trainable
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.output_format = output_format

        # creating kernels for CQT
        Q = float(filter_scale)/(2**(1/bins_per_octave)-1)

        if verbose==True:
            print("Creating CQT kernels ...", end='\r')

        start = time()
        cqt_kernels, self.kernel_width, lenghts, freqs = create_cqt_kernels(Q,
                                                                            sr,
                                                                            fmin,
                                                                            n_bins,
                                                                            bins_per_octave,
                                                                            norm,
                                                                            window,
                                                                            fmax)
        
        self.register_buffer('lenghts', lenghts)
        self.frequencies = freqs
        
        cqt_kernels_real = torch.tensor(cqt_kernels.real).unsqueeze(1)
        cqt_kernels_imag = torch.tensor(cqt_kernels.imag).unsqueeze(1)

        if trainable:
            cqt_kernels_real = torch.nn.Parameter(cqt_kernels_real, requires_grad=trainable)
            cqt_kernels_imag = torch.nn.Parameter(cqt_kernels_imag, requires_grad=trainable)
            self.register_parameter('cqt_kernels_real', cqt_kernels_real)
            self.register_parameter('cqt_kernels_imag', cqt_kernels_imag)
        else:
            self.register_buffer('cqt_kernels_real', cqt_kernels_real)
            self.register_buffer('cqt_kernels_imag', cqt_kernels_imag)

        if verbose==True:
            print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))


    def forward(self,x, output_format=None, normalization_type='librosa'):
        """
        Convert a batch of waveforms to CQT spectrograms.
        
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape

        normalization_type : str
            Type of the normalisation. The possible options are: \n
            'librosa' : the output fits the librosa one \n
            'convolutional' : the output conserves the convolutional inequalities of the wavelet transform:\n
            for all p ϵ [1, inf] \n
                - || CQT ||_p <= || f ||_p || g ||_1 \n
                - || CQT ||_p <= || f ||_1 || g ||_p \n
                - || CQT ||_2 = || f ||_2 || g ||_2 \n
            'wrap' : wraps positive and negative frequencies into positive frequencies. This means that the CQT of a
            sinus (or a cosinus) with a constant amplitude equal to 1 will have the value 1 in the bin corresponding to
            its frequency.
        """
        output_format = output_format or self.output_format
        
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.kernel_width//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.kernel_width//2)

            x = padding(x)

        # CQT
        CQT_real = conv1d(x, self.cqt_kernels_real, stride=self.hop_length)
        CQT_imag = -conv1d(x, self.cqt_kernels_imag, stride=self.hop_length)

        if normalization_type == 'librosa':
            CQT_real *= torch.sqrt(self.lenghts.view(-1, 1))
            CQT_imag *= torch.sqrt(self.lenghts.view(-1, 1))
        elif normalization_type == 'convolutional':
            pass
        elif normalization_type == 'wrap':
            CQT_real *= 2
            CQT_imag *= 2
        else:
            raise ValueError("The normalization_type %r is not part of our current options." % normalization_type)

        if output_format=='Magnitude':
            if self.trainable==False:
                # Getting CQT Amplitude
                CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
            else:
                CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2)+1e-8)
            return CQT

        elif output_format=='Complex':
            return torch.stack((CQT_real,CQT_imag),-1)

        elif output_format=='Phase':
            phase_real = torch.cos(torch.atan2(CQT_imag,CQT_real))
            phase_imag = torch.sin(torch.atan2(CQT_imag,CQT_real))
            return torch.stack((phase_real,phase_imag), -1)

    def forward_manual(self,x):
        """
        Method for debugging
        """
        
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.kernel_width//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.kernel_width//2)

            x = padding(x)

        # CQT
        CQT_real = conv1d(x, self.cqt_kernels_real, stride=self.hop_length)
        CQT_imag = conv1d(x, self.cqt_kernels_imag, stride=self.hop_length)

        # Getting CQT Amplitude
        CQT = torch.sqrt(CQT_real.pow(2)+CQT_imag.pow(2))
        return CQT*torch.sqrt(self.lenghts.view(-1,1))


class CQT2010v2(torch.nn.Module):
    """This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.

    This alogrithm uses the resampling method proposed in [1].
    Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency
    spectrum, we make a small CQT kernel covering only the top octave. Then we keep downsampling the
    input audio by a factor of 2 to convoluting it with the small CQT kernel.
    Everytime the input audio is downsampled, the CQT relative to the downsampled input is equivalent
    to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the
    code from the 1992 alogrithm [2]
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).

    Early downsampling factor is to downsample the input audio to reduce the CQT kernel size.
    The result with and without early downsampling are more or less the same except in the very low
    frequency region where freq < 40Hz.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.  If ``fmax`` is not ``None``, then the
        argument ``n_bins`` will be ignored and ``n_bins`` will be calculated automatically.
        Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    norm : bool
        Normalization for the CQT result.

    basis_norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : str
        The windowing function for CQT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``

    output_format : str
        Determine the return type.
        'Magnitude' will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins, time_steps)``;
        'Complex' will return the STFT result in complex number, shape = ``(num_samples, freq_bins, time_steps, 2)``;
        'Phase' will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.

    Returns
    -------
    spectrogram : torch.tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT2010v2()
    >>> specs = spec_layer(x)
    """

    
# To DO:
# need to deal with the filter and other tensors
    
    def __init__(self, sr=22050, hop_length=512, fmin=32.70, fmax=None, n_bins=84, filter_scale=1,
                bins_per_octave=12, norm=True, basis_norm=1, window='hann', pad_mode='reflect',
                earlydownsample=True, trainable=False, output_format='Magnitude', verbose=True):

        super().__init__()

        self.norm = norm  # Now norm is used to normalize the final CQT result by dividing n_fft
        # basis_norm is for normalizing basis
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins
        self.earlydownsample = earlydownsample  # We will activate early downsampling later if possible
        self.trainable = trainable
        self.output_format = output_format

        # It will be used to calculate filter_cutoff and creating CQT kernels
        Q = float(filter_scale)/(2**(1/bins_per_octave)-1)

        # Creating lowpass filter and make it a torch tensor
        if verbose==True:
            print("Creating low pass filter ...", end='\r')
        start = time()
        # self.lowpass_filter = torch.tensor(
        #                                     create_lowpass_filter(
        #                                     band_center = 0.50,
        #                                     kernelLength=256,
        #                                     transitionBandwidth=0.001))
        lowpass_filter = torch.tensor(create_lowpass_filter(
                                                            band_center = 0.50,
                                                            kernelLength=256,
                                                            transitionBandwidth=0.001)
                                                            )

        # Broadcast the tensor to the shape that fits conv1d
        self.register_buffer('lowpass_filter', lowpass_filter[None,None,:])
        if verbose==True:
            print("Low pass filter created, time used = {:.4f} seconds".format(time()-start))

        # Caluate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
        if verbose==True:
            print("num_octave = ", self.n_octaves)

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = fmin*2**(self.n_octaves-1)
        remainder = n_bins % bins_per_octave
        # print("remainder = ", remainder)

        if remainder==0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t*2**((bins_per_octave-1)/bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t*2**((remainder-1)/bins_per_octave)

        self.fmin_t = fmax_t/2**(1-1/bins_per_octave) # Adjusting the top minium bins
        if fmax_t > sr/2:
            raise ValueError('The top bin {}Hz has exceeded the Nyquist frequency, \
                            please reduce the n_bins'.format(fmax_t))

        if self.earlydownsample == True: # Do early downsampling if this argument is True
            if verbose==True:
                print("Creating early downsampling filter ...", end='\r')
            start = time()
            sr, self.hop_length, self.downsample_factor, early_downsample_filter, \
                self.earlydownsample = get_early_downsample_params(sr,
                                                                        hop_length,
                                                                        fmax_t,
                                                                        Q,
                                                                        self.n_octaves,
                                                                        verbose)
            self.register_buffer('early_downsample_filter', early_downsample_filter)
            
            if verbose==True:
                print("Early downsampling filter created, \
                        time used = {:.4f} seconds".format(time()-start))
        else:
            self.downsample_factor=1.

        # Preparing CQT kernels
        if verbose==True:
            print("Creating CQT kernels ...", end='\r')
        start = time()
        basis, self.n_fft, lenghts, _ = create_cqt_kernels(Q,
                                                             sr,
                                                             self.fmin_t,
                                                             n_filters,
                                                             bins_per_octave,
                                                             norm=basis_norm,
                                                             topbin_check=False)
        # For normalization in the end
        # The freqs returned by create_cqt_kernels cannot be used
        # Since that returns only the top octave bins
        # We need the information for all freq bin
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))
        self.frequencies = freqs
        
        lenghts = np.ceil(Q * sr / freqs)
        lenghts = torch.tensor(lenghts).float()
        self.register_buffer('lenghts', lenghts)

        self.basis = basis
        # These cqt_kernel is already in the frequency domain
        cqt_kernels_real = torch.tensor(basis.real.astype(np.float32)).unsqueeze(1)
        cqt_kernels_imag = torch.tensor(basis.imag.astype(np.float32)).unsqueeze(1)
        
        if trainable:
            cqt_kernels_real = torch.nn.Parameter(cqt_kernels_real, requires_grad=trainable)
            cqt_kernels_imag = torch.nn.Parameter(cqt_kernels_imag, requires_grad=trainable)
            self.register_parameter('cqt_kernels_real', cqt_kernels_real)
            self.register_parameter('cqt_kernels_imag', cqt_kernels_imag)
        else:
            self.register_buffer('cqt_kernels_real', cqt_kernels_real)
            self.register_buffer('cqt_kernels_imag', cqt_kernels_imag)         


        if verbose==True:
            print("CQT kernels created, time used = {:.4f} seconds".format(time()-start))
        # print("Getting cqt kernel done, n_fft = ",self.n_fft)

        # If center==True, the STFT window will be put in the middle, and paddings at the beginning
        # and ending are required.
        if self.pad_mode == 'constant':
            self.padding = nn.ConstantPad1d(self.n_fft//2, 0)
        elif self.pad_mode == 'reflect':
            self.padding = nn.ReflectionPad1d(self.n_fft//2)
            

    def forward(self,x,output_format=None, normalization_type='librosa'):
        """
        Convert a batch of waveforms to CQT spectrograms.
        
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """              
        output_format = output_format or self.output_format
        
        x = broadcast_dim(x)
        if self.earlydownsample==True:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor)
        hop = self.hop_length
        CQT = get_cqt_complex(x, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding)  # Getting the top octave CQT

        x_down = x  # Preparing a new variable for downsampling

        for i in range(self.n_octaves-1):
            hop = hop//2
            x_down = downsampling_by_2(x_down, self.lowpass_filter)
            CQT1 = get_cqt_complex(x_down, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding)
            CQT = torch.cat((CQT1, CQT),1)

        CQT = CQT[:,-self.n_bins:,:]  # Removing unwanted bottom bins
        # print("downsample_factor = ",self.downsample_factor)
        # print(CQT.shape)
        # print(self.lenghts.view(-1,1).shape)

        # Normalizing the output with the downsampling factor, 2**(self.n_octaves-1) is make it
        # same mag as 1992
        CQT = CQT*self.downsample_factor
        # Normalize again to get same result as librosa
        if normalization_type == 'librosa':
            CQT = CQT*torch.sqrt(self.lenghts.view(-1,1,1))
        elif normalization_type == 'convolutional':
            pass
        elif normalization_type == 'wrap':
            CQT *= 2
        else:
            raise ValueError("The normalization_type %r is not part of our current options." % normalization_type)

        

        if output_format=='Magnitude':
            if self.trainable==False:
                # Getting CQT Amplitude
                return torch.sqrt(CQT.pow(2).sum(-1))
            else:
                return torch.sqrt(CQT.pow(2).sum(-1)+1e-8)

        elif output_format=='Complex':
            return CQT

        elif output_format=='Phase':
            phase_real = torch.cos(torch.atan2(CQT[:,:,:,1],CQT[:,:,:,0]))
            phase_imag = torch.sin(torch.atan2(CQT[:,:,:,1],CQT[:,:,:,0]))
            return torch.stack((phase_real,phase_imag), -1)


class CQT(CQT1992v2):
    """An abbreviation for :func:`~nnAudio.Spectrogram.CQT1992v2`. Please refer to the :func:`~nnAudio.Spectrogram.CQT1992v2` documentation"""
    pass



# The section below is for developing purpose
# Please don't use the following classes
#

class DFT(torch.nn.Module):
    """
    Experimental feature before `torch.fft` was made avaliable. 
    The inverse function only works for 1 single frame. i.e. input shape = (batch, n_fft, 1)
    """
    def __init__(self, n_fft=2048, freq_bins=None, hop_length=512,
                window='hann', freq_scale='no', center=True, pad_mode='reflect',
                fmin=50, fmax=6000, sr=22050):

        super().__init__()

        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft

        # Create filter windows for stft
        wsin, wcos, self.bins2freq = create_fourier_kernels(n_fft=n_fft,
                                                            freq_bins=n_fft,
                                                            window=window,
                                                            freq_scale=freq_scale,
                                                            fmin=fmin,
                                                            fmax=fmax,
                                                            sr=sr)
        self.wsin = torch.tensor(wsin, dtype=torch.float)
        self.wcos = torch.tensor(wcos, dtype=torch.float)

    def forward(self,x):
        """
        Convert a batch of waveforms to spectrums.
        
        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """              
        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.n_fft//2, 0)
            elif self.pad_mode == 'reflect':
                padding = nn.ReflectionPad1d(self.n_fft//2)

            x = padding(x)

        imag = conv1d(x, self.wsin, stride=self.stride)
        real = conv1d(x, self.wcos, stride=self.stride)
        return (real, -imag)

    def inverse(self,x_real,x_imag):
        """
        Convert a batch of waveforms to CQT spectrograms.
        
        Parameters
        ----------
        x_real : torch tensor
            Real part of the signal.
        x_imag : torch tensor
            Imaginary part of the signal.
        """              
        x_real = broadcast_dim(x_real)
        x_imag = broadcast_dim(x_imag)

        x_real.transpose_(1,2) # Prepare the right shape to do inverse
        x_imag.transpose_(1,2) # Prepare the right shape to do inverse

        # if self.center:
        #     if self.pad_mode == 'constant':
        #         padding = nn.ConstantPad1d(self.n_fft//2, 0)
        #     elif self.pad_mode == 'reflect':
        #         padding = nn.ReflectionPad1d(self.n_fft//2)

        #     x_real = padding(x_real)
        #     x_imag = padding(x_imag)

        # Watch out for the positive and negative signs
        # ifft = e^(+2\pi*j)*X

        # ifft(X_real) = (a1, a2)

        # ifft(X_imag)*1j = (b1, b2)*1j
        #                = (-b2, b1)

        a1 = conv1d(x_real, self.wcos, stride=self.stride)
        a2 = conv1d(x_real, self.wsin, stride=self.stride)
        b1 = conv1d(x_imag, self.wcos, stride=self.stride)
        b2 = conv1d(x_imag, self.wsin, stride=self.stride)

        imag = a2+b1
        real = a1-b2
        return (real/self.n_fft, imag/self.n_fft)



    
class iSTFT(torch.nn.Module):
    """This class is to convert spectrograms back to waveforms. It only works for the complex value spectrograms.
    If you have the magnitude spectrograms, please use :func:`~nnAudio.Spectrogram.Griffin_Lim`. 
    The parameters (e.g. n_fft, window) need to be the same as the STFT in order to obtain the correct inverse.
    If trainability is not required, it is recommended to use the ``inverse`` method under the ``STFT`` class 
    to save GPU/RAM memory.
    
    When ``trainable=True`` and ``freq_scale!='no'``, there is no guarantee that the inverse is perfect, please
    use with extra care.

    Parameters
    ----------
    n_fft : int
        The window size. Default value is 2048.

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins
        Please make sure the value is the same as the forward STFT.

    hop_length : int
        The hop (or stride) size. Default value is ``None`` which is equivalent to ``n_fft//4``.
        Please make sure the value is the same as the forward STFT.

    window : str
        The windowing function for iSTFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.
        Please make sure the value is the same as the forward STFT.

    freq_scale : 'linear', 'log', or 'no'
        Determine the spacing between each frequency bin. When `linear` or `log` is used,
        the bin spacing can be controlled by ``fmin`` and ``fmax``. If 'no' is used, the bin will
        start at 0Hz and end at Nyquist frequency with linear spacing.
        Please make sure the value is the same as the forward STFT.

    center : bool
        Putting the iSTFT keneral at the center of the time-step or not. If ``False``, the time
        index is the beginning of the iSTFT kernel, if ``True``, the time index is the center of
        the iSTFT kernel. Default value if ``True``. 
        Please make sure the value is the same as the forward STFT.

    fmin : int
        The starting frequency for the lowest frequency bin. If freq_scale is ``no``, this argument
        does nothing. Please make sure the value is the same as the forward STFT.

    fmax : int
        The ending frequency for the highest frequency bin. If freq_scale is ``no``, this argument
        does nothing. Please make sure the value is the same as the forward STFT.

    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    trainable_kernels : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT
        kernels will also be caluclated and the STFT kernels will be updated during model training.
        Default value is ``False``.
        
    trainable_window : bool
        Determine if the window function is trainable or not.
        Default value is ``False``.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.

    Returns
    -------
    spectrogram : torch.tensor
        It returns a batch of waveforms.

    Examples
    --------
    >>> spec_layer = Spectrogram.iSTFT()
    >>> specs = spec_layer(x)
    """

    def __init__(self, n_fft=2048, win_length=None, freq_bins=None, hop_length=None, window='hann',
                freq_scale='no', center=True, fmin=50, fmax=6000, sr=22050, trainable_kernels=False,
                trainable_window=False, verbose=True, refresh_win=True):

        super().__init__()

        # Trying to make the default setting same as librosa
        if win_length==None: win_length = n_fft
        if hop_length==None: hop_length = int(win_length // 4)

        self.n_fft = n_fft    
        self.win_length = win_length
        self.stride = hop_length
        self.center = center
        
        self.pad_amount = self.n_fft // 2
        self.refresh_win = refresh_win 
        
        start = time()

        # Create the window function and prepare the shape for batch-wise-time-wise multiplication

        # Create filter windows for inverse
        kernel_sin, kernel_cos, _, _, window_mask = create_fourier_kernels(n_fft,
                                                          win_length=win_length,
                                                          freq_bins=n_fft,
                                                          window=window,
                                                          freq_scale=freq_scale,
                                                          fmin=fmin,
                                                          fmax=fmax,
                                                          sr=sr,
                                                          verbose=False)
        window_mask = get_window(window,int(win_length), fftbins=True)
        
        # For inverse, the Fourier kernels do not need to be windowed
        window_mask = torch.tensor(window_mask).unsqueeze(0).unsqueeze(-1)
        
        # kernel_sin and kernel_cos have the shape (freq_bins, 1, n_fft, 1) to support 2D Conv
        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float).unsqueeze(-1)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float).unsqueeze(-1)        
        
        # Decide if the Fourier kernels are trainable
        if trainable_kernels:
            # Making all these variables trainable
            kernel_sin = torch.nn.Parameter(kernel_sin, requires_grad=trainable_kernels)
            kernel_cos = torch.nn.Parameter(kernel_cos, requires_grad=trainable_kernels)
            self.register_parameter('kernel_sin', kernel_sin)
            self.register_parameter('kernel_cos', kernel_cos)

        else:
            self.register_buffer('kernel_sin', kernel_sin)
            self.register_buffer('kernel_cos', kernel_cos)
            
        # Decide if the window function is trainable
        if trainable_window:
            window_mask = torch.nn.Parameter(window_mask, requires_grad=trainable_window)
            self.register_parameter('window_mask', window_mask)
        else:
            self.register_buffer('window_mask', window_mask)
        

        if verbose==True:
            print("iSTFT kernels created, time used = {:.4f} seconds".format(time()-start))
        else:
            pass


    def forward(self, X, onesided=False, length=None, refresh_win=None):
        """
        If your spectrograms only have ``n_fft//2+1`` frequency bins, please use ``onesided=True``,
        else use ``onesided=False``
        To make sure the inverse STFT has the same output length of the original waveform, please
        set `length` as your intended waveform length. By default, ``length=None``,
        which will remove ``n_fft//2`` samples from the start and the end of the output.
        If your input spectrograms X are of the same length, please use ``refresh_win=None`` to increase
        computational speed.
        """
        if refresh_win==None:
            refresh_win=self.refresh_win
                
        assert X.dim()==4 , "Inverse iSTFT only works for complex number," \
                            "make sure our tensor is in the shape of (batch, freq_bins, timesteps, 2)" 
        
        # If the input spectrogram contains only half of the n_fft
        # Use extend_fbins function to get back another half
        if onesided:
            X = extend_fbins(X) # extend freq

    
        X_real, X_imag = X[:, :, :, 0], X[:, :, :, 1]

        # broadcast dimensions to support 2D convolution
        X_real_bc = X_real.unsqueeze(1)
        X_imag_bc = X_imag.unsqueeze(1)
        
        a1 = conv2d(X_real_bc, self.kernel_cos, stride=(1,1))
        b2 = conv2d(X_imag_bc, self.kernel_sin, stride=(1,1))
       
        # compute real and imag part. signal lies in the real part
        real = a1 - b2
        real = real.squeeze(-2)*self.window_mask

        # Normalize the amplitude with n_fft
        real /= (self.n_fft)

        # Overlap and Add algorithm to connect all the frames
        real = overlap_add(real, self.stride)
    
        # Prepare the window sumsqure for division
        # Only need to create this window once to save time
        # Unless the input spectrograms have different time steps
        if hasattr(self, 'w_sum')==False or refresh_win==True:
            self.w_sum = torch_window_sumsquare(self.window_mask.flatten(), X.shape[2], self.stride, self.n_fft).flatten()
            self.nonzero_indices = (self.w_sum>1e-10)    
        else:
            pass
        real[:, self.nonzero_indices] = real[:,self.nonzero_indices].div(self.w_sum[self.nonzero_indices])
        # Remove padding
        if length is None:       
            if self.center:
                real = real[:, self.pad_amount:-self.pad_amount]

        else:
            if self.center:
                real = real[:, self.pad_amount:self.pad_amount + length]    
            else:
                real = real[:, :length] 
            
        return real
    
    
class Griffin_Lim(torch.nn.Module):
    """
    Converting Magnitude spectrograms back to waveforms based on the "fast Griffin-Lim"[1].
    This Griffin Lim is a direct clone from librosa.griffinlim.
    
    [1] Perraudin, N., Balazs, P., & Søndergaard, P. L. “A fast Griffin-Lim algorithm,”
    IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4), Oct. 2013.
    
    Parameters
    ----------
    n_fft : int
        The window size. Default value is 2048.

    n_iter=32 : int
        The number of iterations for Griffin-Lim. The default value is ``32``

    hop_length : int
        The hop (or stride) size. Default value is ``None`` which is equivalent to ``n_fft//4``.
        Please make sure the value is the same as the forward STFT.

    window : str
        The windowing function for iSTFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.
        Please make sure the value is the same as the forward STFT.

    center : bool
        Putting the iSTFT keneral at the center of the time-step or not. If ``False``, the time
        index is the beginning of the iSTFT kernel, if ``True``, the time index is the center of
        the iSTFT kernel. Default value if ``True``. 
        Please make sure the value is the same as the forward STFT.

    momentum : float
        The momentum for the update rule. The default value is ``0.99``.

    device : str
        Choose which device to initialize this layer. Default value is 'cpu'
    
    """
    
    def __init__(self,
                 n_fft,
                 n_iter=32,
                 hop_length=None,
                 win_length=None,
                 window='hann', 
                 center=True,
                 pad_mode='reflect',
                 momentum=0.99,
                 device='cpu'):
        super().__init__()
        
        self.n_fft = n_fft
        self.win_length = win_length
        self.n_iter = n_iter
        self.center = center
        self.pad_mode = pad_mode
        self.momentum = momentum
        self.device = device
        if win_length==None:
            self.win_length=n_fft
        else:
            self.win_length=win_length
        if hop_length==None:
            self.hop_length = n_fft//4
        else:
            self.hop_length = hop_length
            
        # Creating window function for stft and istft later
        self.w = torch.tensor(get_window(window,
                                         int(self.win_length), 
                                         fftbins=True), 
                              device=device).float()

    def forward(self, S):
        """
        Convert a batch of magnitude spectrograms to waveforms.
        
        Parameters
        ----------
        S : torch tensor
            Spectrogram of the shape ``(batch, n_fft//2+1, timesteps)``    
        """
        
        assert S.dim()==3 , "Please make sure your input is in the shape of (batch, freq_bins, timesteps)"
        
        # Initializing Random Phase
        rand_phase = torch.randn(*S.shape, device=self.device)
        angles = torch.empty((*S.shape,2), device=self.device)
        angles[:, :,:,0] = torch.cos(2 * np.pi * rand_phase)
        angles[:,:,:,1] = torch.sin(2 * np.pi * rand_phase)
        
        # Initializing the rebuilt magnitude spectrogram
        rebuilt = torch.zeros(*angles.shape, device=self.device)
        
        for _ in range(self.n_iter):
            tprev = rebuilt # Saving previous rebuilt magnitude spec

            # spec2wav conversion
#             print(f'win_length={self.win_length}\tw={self.w.shape}')
            inverse = torch.istft(S.unsqueeze(-1) * angles,
                                  self.n_fft,
                                  self.hop_length,
                                  win_length=self.win_length, 
                                  window=self.w,
                                  center=self.center)
            # wav2spec conversion
            rebuilt = torch.stft(inverse,
                                 self.n_fft,
                                 self.hop_length,
                                 win_length=self.win_length,
                                 window=self.w,
                                 pad_mode=self.pad_mode)

            # Phase update rule
            angles[:,:,:] = rebuilt[:,:,:] - (self.momentum / (1 + self.momentum)) * tprev[:,:,:]

            # Phase normalization
            angles = angles.div(torch.sqrt(angles.pow(2).sum(-1)).unsqueeze(-1) + 1e-16) # normalizing the phase
        
        # Using the final phase to reconstruct the waveforms
        inverse = torch.istft(S.unsqueeze(-1) * angles,
                              self.n_fft,
                              self.hop_length,
                              win_length=self.win_length, 
                              window=self.w,
                              center=self.center)
        return inverse



class Combined_Frequency_Periodicity(nn.Module):
    """
    Vectorized version of the code in https://github.com/leo-so/VocalMelodyExtPatchCNN/blob/master/MelodyExt.py.
    This feature is described in 'Combining Spectral and Temporal Representations for Multipitch Estimation of Polyphonic Music'
    https://ieeexplore.ieee.org/document/7118691
    
    Under development, please report any bugs you found
    """
    def __init__(self,fr=2, fs=16000, hop_length=320,
                 window_size=2049, fc=80, tc=1/1000,
                 g=[0.24, 0.6, 1], NumPerOct=48):
        super().__init__()
        
        self.window_size = window_size
        self.hop_length = hop_length
        
        # variables for STFT part
        self.N = int(fs/float(fr)) # Will be used to calculate padding
        self.f = fs*np.linspace(0, 0.5, np.round(self.N//2), endpoint=True) # it won't be used but will be returned          
        self.pad_value = ((self.N-window_size))
        # Create window function, always blackmanharris?
        h = scipy.signal.blackmanharris(window_size).astype(np.float32) # window function for STFT
        self.register_buffer('h',torch.tensor(h))
        
        # variables for CFP
        self.NumofLayer = np.size(g)
        self.g = g
        self.tc_idx = round(fs*tc) # index to filter out top tc_idx and bottom tc_idx bins
        self.fc_idx = round(fc/fr) # index to filter out top fc_idx and bottom fc_idx bins
        self.HighFreqIdx = int(round((1/tc)/fr)+1)
        self.HighQuefIdx = int(round(fs/fc)+1)
        
        # attributes to be returned
        self.f = self.f[:self.HighFreqIdx]
        self.q = np.arange(self.HighQuefIdx)/float(fs) 
        
        # filters for the final step
        freq2logfreq_matrix, quef2logfreq_matrix = self.create_logfreq_matrix(self.f, self.q, fr, fc, tc, NumPerOct, fs)
        self.register_buffer('freq2logfreq_matrix',torch.tensor(freq2logfreq_matrix.astype(np.float32)))
        self.register_buffer('quef2logfreq_matrix',torch.tensor(quef2logfreq_matrix.astype(np.float32)))
    
    def _CFP(self, spec):
        spec = torch.relu(spec).pow(self.g[0])
        
        if self.NumofLayer >= 2:
            for gc in range(1, self.NumofLayer):
                if np.remainder(gc, 2) == 1:
                    ceps = torch.rfft(spec, 1, onesided=False)[:,:,:,0]/np.sqrt(self.N)
                    ceps = self.nonlinear_func(ceps, self.g[gc], self.tc_idx)
                else:
                    spec = torch.rfft(ceps, 1, onesided=False)[:,:,:,0]/np.sqrt(self.N)
                    spec = self.nonlinear_func(spec, self.g[gc], self.fc_idx)    
        
        return spec, ceps
    
    
    def forward(self, x):
        tfr0 = torch.stft(x, self.N, hop_length=self.hop_length, win_length=self.window_size,
                   window=self.h, onesided=False, pad_mode='constant')
        tfr0 = torch.sqrt(tfr0.pow(2).sum(-1))/torch.norm(self.h) # calcuate magnitude
        tfr0 = tfr0.transpose(1,2)[:,1:-1] #transpose F and T axis and discard first and last frames
        # The transpose is necessary for rfft later
        # (batch, timesteps, n_fft)
        tfr, ceps = self._CFP(tfr0)
        
#         return tfr0
        # removing duplicate bins
        tfr0 = tfr0[:,:,:int(round(self.N/2))]
        tfr = tfr[:,:,:int(round(self.N/2))]
        ceps = ceps[:,:,:int(round(self.N/2))]       

        # Crop up to the highest frequency
        tfr0 = tfr0[:,:,:self.HighFreqIdx]
        tfr = tfr[:,:,:self.HighFreqIdx]
        ceps = ceps[:,:,:self.HighQuefIdx]       
        tfrL0 = torch.matmul(self.freq2logfreq_matrix, tfr0.transpose(1,2))
        tfrLF = torch.matmul(self.freq2logfreq_matrix, tfr.transpose(1,2))
        tfrLQ = torch.matmul(self.quef2logfreq_matrix, ceps.transpose(1,2))
        Z = tfrLF * tfrLQ
        
        # Only need to calculate this once
        self.t = np.arange(self.hop_length,
                           np.ceil(len(x)/float(self.hop_length))*self.hop_length,
                           self.hop_length) # it won't be used but will be returned
        
        return Z, tfrL0, tfrLF, tfrLQ                    
                    
    def nonlinear_func(self, X, g, cutoff):
        cutoff = int(cutoff)
        if g!=0:
            X = torch.relu(X)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
            X = X.pow(g)
        else: # when g=0, it converges to log
            X = torch.log(X)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
        return X   
    
    def create_logfreq_matrix(self, f, q, fr, fc, tc, NumPerOct, fs):
        StartFreq = fc
        StopFreq = 1/tc
        Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
        central_freq = [] # A list holding the frequencies in log scale

        for i in range(0, Nest):
            CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
            if CenFreq < StopFreq:
                central_freq.append(CenFreq)
            else:
                break

        Nest = len(central_freq)    
        freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)

        # Calculating the freq_band_transformation 
        for i in range(1, Nest-1):
            l = int(round(central_freq[i-1]/fr))
            r = int(round(central_freq[i+1]/fr)+1)
            #rounding1
            if l >= r-1:
                freq_band_transformation[i, l] = 1
            else:
                for j in range(l, r):
                    if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                        freq_band_transformation[i, j] = (f[j] - central_freq[i-1]) / (central_freq[i] - central_freq[i-1])
                    elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                        freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])

        # Calculating the quef_band_transformation                
        f = 1/q # divide by 0, do I need to fix this?   
        quef_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
        for i in range(1, Nest-1):      
            for j in range(int(round(fs/central_freq[i+1])), int(round(fs/central_freq[i-1])+1)):
                if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                    quef_band_transformation[i, j] = (f[j] - central_freq[i-1])/(central_freq[i] - central_freq[i-1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                    quef_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])                    
                        
        return freq_band_transformation, quef_band_transformation
    
    
class CFP(nn.Module):
    """
    This is the modified version so that the number of timesteps fits with other classes
    
    Under development, please report any bugs you found
    """
    def __init__(self,fr=2, fs=16000, hop_length=320,
                 window_size=2049, fc=80, tc=1/1000,
                 g=[0.24, 0.6, 1], NumPerOct=48):
        super().__init__()
        
        self.window_size = window_size
        self.hop_length = hop_length
        
        # variables for STFT part
        self.N = int(fs/float(fr)) # Will be used to calculate padding
        self.f = fs*np.linspace(0, 0.5, np.round(self.N//2), endpoint=True) # it won't be used but will be returned          
        self.pad_value = ((self.N-window_size))
        # Create window function, always blackmanharris?
        h = scipy.signal.blackmanharris(window_size).astype(np.float32) # window function for STFT
        self.register_buffer('h',torch.tensor(h))
        
        # variables for CFP
        self.NumofLayer = np.size(g)
        self.g = g
        self.tc_idx = round(fs*tc) # index to filter out top tc_idx and bottom tc_idx bins
        self.fc_idx = round(fc/fr) # index to filter out top fc_idx and bottom fc_idx bins
        self.HighFreqIdx = int(round((1/tc)/fr)+1)
        self.HighQuefIdx = int(round(fs/fc)+1)
        
        # attributes to be returned
        self.f = self.f[:self.HighFreqIdx]
        self.q = np.arange(self.HighQuefIdx)/float(fs) 
        
        # filters for the final step
        freq2logfreq_matrix, quef2logfreq_matrix = self.create_logfreq_matrix(self.f, self.q, fr, fc, tc, NumPerOct, fs)
        self.register_buffer('freq2logfreq_matrix',torch.tensor(freq2logfreq_matrix.astype(np.float32)))
        self.register_buffer('quef2logfreq_matrix',torch.tensor(quef2logfreq_matrix.astype(np.float32)))
    
    def _CFP(self, spec):
        spec = torch.relu(spec).pow(self.g[0])
        
        if self.NumofLayer >= 2:
            for gc in range(1, self.NumofLayer):
                if np.remainder(gc, 2) == 1:
                    ceps = torch.rfft(spec, 1, onesided=False)[:,:,:,0]/np.sqrt(self.N)
                    ceps = self.nonlinear_func(ceps, self.g[gc], self.tc_idx)
                else:
                    spec = torch.rfft(ceps, 1, onesided=False)[:,:,:,0]/np.sqrt(self.N)
                    spec = self.nonlinear_func(spec, self.g[gc], self.fc_idx)    
        
        return spec, ceps
    
    
    def forward(self, x):
        tfr0 = torch.stft(x, self.N, hop_length=self.hop_length, win_length=self.window_size,
                   window=self.h, onesided=False, pad_mode='constant')
        tfr0 = torch.sqrt(tfr0.pow(2).sum(-1))/torch.norm(self.h) # calcuate magnitude
        tfr0 = tfr0.transpose(1,2) #transpose F and T axis and discard first and last frames
        # The transpose is necessary for rfft later
        # (batch, timesteps, n_fft)
        tfr, ceps = self._CFP(tfr0)
        
#         return tfr0
        # removing duplicate bins
        tfr0 = tfr0[:,:,:int(round(self.N/2))]
        tfr = tfr[:,:,:int(round(self.N/2))]
        ceps = ceps[:,:,:int(round(self.N/2))]       

        # Crop up to the highest frequency
        tfr0 = tfr0[:,:,:self.HighFreqIdx]
        tfr = tfr[:,:,:self.HighFreqIdx]
        ceps = ceps[:,:,:self.HighQuefIdx]       
        tfrL0 = torch.matmul(self.freq2logfreq_matrix, tfr0.transpose(1,2))
        tfrLF = torch.matmul(self.freq2logfreq_matrix, tfr.transpose(1,2))
        tfrLQ = torch.matmul(self.quef2logfreq_matrix, ceps.transpose(1,2))
        Z = tfrLF * tfrLQ
        
        # Only need to calculate this once
        self.t = np.arange(self.hop_length,
                           np.ceil(len(x)/float(self.hop_length))*self.hop_length,
                           self.hop_length) # it won't be used but will be returned
        
        return Z#, tfrL0, tfrLF, tfrLQ                    
                    
    def nonlinear_func(self, X, g, cutoff):
        cutoff = int(cutoff)
        if g!=0:
            X = torch.relu(X)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
            X = X.pow(g)
        else: # when g=0, it converges to log
            X = torch.log(X)
            X[:, :, :cutoff] = 0
            X[:, :, -cutoff:] = 0
        return X   
    
    def create_logfreq_matrix(self, f, q, fr, fc, tc, NumPerOct, fs):
        StartFreq = fc
        StopFreq = 1/tc
        Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
        central_freq = [] # A list holding the frequencies in log scale

        for i in range(0, Nest):
            CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
            if CenFreq < StopFreq:
                central_freq.append(CenFreq)
            else:
                break

        Nest = len(central_freq)    
        freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)

        # Calculating the freq_band_transformation 
        for i in range(1, Nest-1):
            l = int(round(central_freq[i-1]/fr))
            r = int(round(central_freq[i+1]/fr)+1)
            #rounding1
            if l >= r-1:
                freq_band_transformation[i, l] = 1
            else:
                for j in range(l, r):
                    if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                        freq_band_transformation[i, j] = (f[j] - central_freq[i-1]) / (central_freq[i] - central_freq[i-1])
                    elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                        freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])

        # Calculating the quef_band_transformation                
        f = 1/q # divide by 0, do I need to fix this?   
        quef_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
        for i in range(1, Nest-1):      
            for j in range(int(round(fs/central_freq[i+1])), int(round(fs/central_freq[i-1])+1)):
                if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                    quef_band_transformation[i, j] = (f[j] - central_freq[i-1])/(central_freq[i] - central_freq[i-1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                    quef_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])                    
                        
        return freq_band_transformation, quef_band_transformation    
