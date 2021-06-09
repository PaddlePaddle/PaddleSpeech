"""
Module containing functions cloned from librosa

To make sure nnAudio would not become broken when updating librosa 
"""

import numpy as np
import warnings
### ----------------Functions for generating kenral for Mel Spectrogram------------ ###
# This code is equalvant to from librosa.filters import mel
# By doing so, we can run nnAudio without installing librosa
def fft2gammatonemx(sr=20000, n_fft=2048, n_bins=64, width=1.0, fmin=0.0,
                    fmax=11025, maxlen=1024):
    """
    # Ellis' description in MATLAB:
    # [wts,cfreqa] = fft2gammatonemx(nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
    #      Generate a matrix of weights to combine FFT bins into
    #      Gammatone bins.  nfft defines the source FFT size at
    #      sampling rate sr.  Optional nfilts specifies the number of
    #      output bands required (default 64), and width is the
    #      constant width of each band in Bark (default 1).
    #      minfreq, maxfreq specify range covered in Hz (100, sr/2).
    #      While wts has nfft columns, the second half are all zero.
    #      Hence, aud spectrum is
    #      fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft));
    #      maxlen truncates the rows to this many bins.
    #      cfreqs returns the actual center frequencies of each
    #      gammatone band in Hz.
    #
    # 2009/02/22 02:29:25 Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    # Sat May 27 15:37:50 2017 Maddie Cusimano, mcusi@mit.edu 27 May 2017: convert to python
    """

    wts = np.zeros([n_bins, n_fft], dtype=np.float32)

    # after Slaney's MakeERBFilters
    EarQ = 9.26449;
    minBW = 24.7;
    order = 1;

    nFr = np.array(range(n_bins)) + 1
    em = EarQ * minBW
    cfreqs = (fmax + em) * np.exp(nFr * (-np.log(fmax + em) + np.log(fmin + em)) / n_bins) - em
    cfreqs = cfreqs[::-1]

    GTord = 4
    ucircArray = np.array(range(int(n_fft / 2 + 1)))
    ucirc = np.exp(1j * 2 * np.pi * ucircArray / n_fft);
    # justpoles = 0 :taking out the 'if' corresponding to this.

    ERB = width * np.power(np.power(cfreqs / EarQ, order) + np.power(minBW, order), 1 / order);
    B = 1.019 * 2 * np.pi * ERB;
    r = np.exp(-B / sr)
    theta = 2 * np.pi * cfreqs / sr
    pole = r * np.exp(1j * theta)
    T = 1 / sr
    ebt = np.exp(B * T);
    cpt = 2 * cfreqs * np.pi * T;
    ccpt = 2 * T * np.cos(cpt);
    scpt = 2 * T * np.sin(cpt);
    A11 = -np.divide(np.divide(ccpt, ebt) + np.divide(np.sqrt(3 + 2 ** 1.5) * scpt, ebt), 2);
    A12 = -np.divide(np.divide(ccpt, ebt) - np.divide(np.sqrt(3 + 2 ** 1.5) * scpt, ebt), 2);
    A13 = -np.divide(np.divide(ccpt, ebt) + np.divide(np.sqrt(3 - 2 ** 1.5) * scpt, ebt), 2);
    A14 = -np.divide(np.divide(ccpt, ebt) - np.divide(np.sqrt(3 - 2 ** 1.5) * scpt, ebt), 2);
    zros = -np.array([A11, A12, A13, A14]) / T;
    wIdx = range(int(n_fft / 2 + 1))
    gain = np.abs((-2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T + 2 * np.exp(
        -(B * T) + 2 * 1j * cfreqs * np.pi * T) * T * (
                               np.cos(2 * cfreqs * np.pi * T) - np.sqrt(3 - 2 ** (3 / 2)) * np.sin(
                           2 * cfreqs * np.pi * T))) * (-2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T + 2 * np.exp(
        -(B * T) + 2 * 1j * cfreqs * np.pi * T) * T * (np.cos(2 * cfreqs * np.pi * T) + np.sqrt(
        3 - 2 ** (3 / 2)) * np.sin(2 * cfreqs * np.pi * T))) * (
                              -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T + 2 * np.exp(
                          -(B * T) + 2 * 1j * cfreqs * np.pi * T) * T * (
                                          np.cos(2 * cfreqs * np.pi * T) - np.sqrt(3 + 2 ** (3 / 2)) * np.sin(
                                      2 * cfreqs * np.pi * T))) * (
                              -2 * np.exp(4 * 1j * cfreqs * np.pi * T) * T + 2 * np.exp(
                          -(B * T) + 2 * 1j * cfreqs * np.pi * T) * T * (
                                          np.cos(2 * cfreqs * np.pi * T) + np.sqrt(3 + 2 ** (3 / 2)) * np.sin(
                                      2 * cfreqs * np.pi * T))) / (
                              -2 / np.exp(2 * B * T) - 2 * np.exp(4 * 1j * cfreqs * np.pi * T) + 2 * (
                                  1 + np.exp(4 * 1j * cfreqs * np.pi * T)) / np.exp(B * T)) ** 4);
    # in MATLAB, there used to be 64 where here it says n_bins:
    wts[:, wIdx] = ((T ** 4) / np.reshape(gain, (n_bins, 1))) * np.abs(
        ucirc - np.reshape(zros[0], (n_bins, 1))) * np.abs(ucirc - np.reshape(zros[1], (n_bins, 1))) * np.abs(
        ucirc - np.reshape(zros[2], (n_bins, 1))) * np.abs(ucirc - np.reshape(zros[3], (n_bins, 1))) * (np.abs(
        np.power(np.multiply(np.reshape(pole, (n_bins, 1)) - ucirc, np.conj(np.reshape(pole, (n_bins, 1))) - ucirc),
                 -GTord)));
    wts = wts[:, range(maxlen)];

    return wts, cfreqs

def gammatone(sr, n_fft, n_bins=64, fmin=20.0, fmax=None, htk=False,
        norm=1, dtype=np.float32):
    """Create a Filterbank matrix to combine FFT bins into Gammatone bins
    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal
    n_fft     : int > 0 [scalar]
        number of FFT components
    n_bins    : int > 0 [scalar]
        number of Mel bands to generate
    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`
    htk       : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization).  Otherwise, leave all the triangles aiming for
        a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.
    Returns
    -------
    G         : np.ndarray [shape=(n_bins, 1 + n_fft/2)]
        Gammatone transform matrix
    """

    if fmax is None:
        fmax = float(sr) / 2
    n_bins = int(n_bins)
    
    weights,_ = fft2gammatonemx(sr=sr, n_fft=n_fft, n_bins=n_bins, fmin=fmin, fmax=fmax, maxlen=int(n_fft//2+1))

    return (1/n_fft)*weights

def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies
    Examples
    --------
    >>> librosa.mel_to_hz(3)
    200.
    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])
    Parameters
    ----------
    mels          : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk           : bool
        use HTK formula instead of Slaney
    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        input mels in Hz
    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs

def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels
    Examples
    --------
    >>> librosa.hz_to_mel(60)
    0.9
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])
    Parameters
    ----------
    frequencies   : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk           : bool
        use HTK formula instead of Slaney
    Returns
    -------
    mels        : number or np.ndarray [shape=(n,)]
        input frequencies in Mels
    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels

def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of `np.fft.fftfreq`
    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate
    n_fft : int > 0 [scalar]
        FFT window size
    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`
    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])
    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)

def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """
    This function is cloned from librosa 0.7.
    Please refer to the original 
    `documentation <https://librosa.org/doc/latest/generated/librosa.mel_frequencies.html?highlight=mel_frequencies#librosa.mel_frequencies>`__
    for more info.
    
    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        Number of mel bins.

    fmin      : float >= 0 [scalar]
        Minimum frequency (Hz).

    fmax      : float >= 0 [scalar]
        Maximum frequency (Hz).

    htk       : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of n_mels frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])
    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)

def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm=1, dtype=np.float32):
    """
    This function is cloned from librosa 0.7.
    Please refer to the original 
    `documentation <https://librosa.org/doc/latest/generated/librosa.filters.mel.html>`__
    for more info.    
    Create a Filterbank matrix to combine FFT bins into Mel-frequency bins


    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal
    n_fft     : int > 0 [scalar]
        number of FFT components
    n_mels    : int > 0 [scalar]
        number of Mel bands to generate
    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`
    htk       : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization).  Otherwise, leave all the triangles aiming for
        a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    Notes
    -----
    This function caches at level 10.
    
    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])
    Clip the maximum frequency to 8KHz
    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(melfb, x_axis='linear')
    >>> plt.ylabel('Mel filter')
    >>> plt.title('Mel filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    >>> plt.show()
    """

    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights
### ------------------End of Functions for generating kenral for Mel Spectrogram ----------------###


### ------------------Functions for making STFT same as librosa ---------------------------------###
def pad_center(data, size, axis=-1, **kwargs):
    '''Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered

    size : int >= len(data) [scalar]
        Length to pad `data`

    axis : int
        Axis along which to pad and center the data

    kwargs : additional keyword arguments
      arguments passed to `np.pad()`

    Returns
    -------
    data_padded : np.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    Raises
    ------
    ParameterError
        If `size < data.shape[axis]`

    See Also
    --------
    numpy.pad
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be '
                              'at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)

### ------------------End of functions for making STFT same as librosa ---------------------------###
