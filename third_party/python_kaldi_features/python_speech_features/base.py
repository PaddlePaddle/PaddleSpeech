# calculate filterbank features. Provides e.g. fbank and mfcc features for use in ASR applications
# Author: James Lyons 2012
from __future__ import division
import numpy
from python_speech_features import sigproc
from scipy.fftpack import dct


def mfcc(signal,
         samplerate=16000,
         winlen=0.025,
         winstep=0.01,
         numcep=13,
         nfilt=23,
         nfft=512,
         lowfreq=20,
         highfreq=None,
         dither=1.0,
         remove_dc_offset=True,
         preemph=0.97,
         ceplifter=22,
         useEnergy=True,
         wintype='povey'):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft,
                         lowfreq, highfreq, dither, remove_dc_offset, preemph,
                         wintype)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)
    if useEnergy:
        feat[:, 0] = numpy.log(
            energy
        )  # replace first cepstral coefficient with log of frame energy
    return feat


def fbank(signal,
          samplerate=16000,
          winlen=0.025,
          winstep=0.01,
          nfilt=40,
          nfft=512,
          lowfreq=0,
          highfreq=None,
          dither=1.0,
          remove_dc_offset=True,
          preemph=0.97,
          wintype='hamming'):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
     winfunc=lambda x:numpy.ones((x,))   
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq = highfreq or samplerate / 2
    frames, raw_frames = sigproc.framesig(signal, winlen * samplerate,
                                          winstep * samplerate, dither, preemph,
                                          remove_dc_offset, wintype)
    pspec = sigproc.powspec(frames, nfft)  # nearly the same until this part
    energy = numpy.sum(raw_frames**2,
                       1)  # this stores the raw energy in each frame
    energy = numpy.where(energy == 0,
                         numpy.finfo(float).eps,
                         energy)  # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
    feat = numpy.where(feat == 0,
                       numpy.finfo(float).eps,
                       feat)  # if feat is zero, we get problems with log

    return feat, energy


def logfbank(signal,
             samplerate=16000,
             winlen=0.025,
             winstep=0.01,
             nfilt=40,
             nfft=512,
             lowfreq=64,
             highfreq=None,
             dither=1.0,
             remove_dc_offset=True,
             preemph=0.97,
             wintype='hamming'):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft,
                         lowfreq, highfreq, dither, remove_dc_offset, preemph,
                         wintype)
    return numpy.log(feat)


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 1127 * numpy.log(1 + hz / 700.0)


def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (numpy.exp(mel / 1127.0) - 1)


def get_filterbanks(nfilt=26,
                    nfft=512,
                    samplerate=16000,
                    lowfreq=0,
                    highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)

    # check kaldi/src/feat/Mel-computations.h
    fbank = numpy.zeros([nfilt, nfft // 2 + 1])
    mel_freq_delta = (highmel - lowmel) / (nfilt + 1)
    for j in range(0, nfilt):
        leftmel = lowmel + j * mel_freq_delta
        centermel = lowmel + (j + 1) * mel_freq_delta
        rightmel = lowmel + (j + 2) * mel_freq_delta
        for i in range(0, nfft // 2):
            mel = hz2mel(i * samplerate / nfft)
            if mel > leftmel and mel < rightmel:
                if mel < centermel:
                    fbank[j, i] = (mel - leftmel) / (centermel - leftmel)
                else:
                    fbank[j, i] = (rightmel - mel) / (rightmel - centermel)
    return fbank


def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes, ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L / 2.) * numpy.sin(numpy.pi * n / L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def delta(feat, N):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N + 1)])
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((N, N), (0, 0)),
                       mode='edge')  # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = numpy.dot(
            numpy.arange(-N, N + 1),
            padded[t:t + 2 * N +
                   1]) / denominator  # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat
