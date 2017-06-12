from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from data_utils import utils
from data_utils.audio import AudioSegment


class AudioFeaturizer(object):
    def __init__(self,
                 specgram_type='linear',
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_freq=None,
                 random_seed=0):
        self._specgram_type = specgram_type
        self._stride_ms = stride_ms
        self._window_ms = window_ms
        self._max_freq = max_freq

    def featurize(self, audio_segment):
        return self._compute_specgram(audio_segment.samples,
                                      audio_segment.sample_rate)

    def _compute_specgram(self, samples, sample_rate):
        if self._specgram_type == 'linear':
            return self._compute_linear_specgram(
                samples, sample_rate, self._stride_ms, self._window_ms,
                self._max_freq)
        else:
            raise ValueError("Unknown specgram_type %s. "
                             "Supported values: linear." % self._specgram_type)

    def _compute_linear_specgram(self,
                                 samples,
                                 sample_rate,
                                 stride_ms=10.0,
                                 window_ms=20.0,
                                 max_freq=None,
                                 eps=1e-14):
        """Laod audio data and calculate the log of spectrogram by FFT.
        Refer to utils.py in https://github.com/baidu-research/ba-dls-deepspeech
        """
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must be greater than half of "
                             "sample rate.")
        if stride_ms > window_ms:
            raise ValueError("Stride size must not be greater than "
                             "window size.")
        stride_size = int(0.001 * sample_rate * stride_ms)
        window_size = int(0.001 * sample_rate * window_ms)
        specgram, freqs = self._specgram_real(
            samples,
            window_size=window_size,
            stride_size=stride_size,
            sample_rate=sample_rate)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
        return np.log(specgram[:ind, :] + eps)

    def _specgram_real(self, samples, window_size, stride_size, sample_rate):
        """Compute the spectrogram by FFT for a discrete real signal.
        Refer to utils.py in https://github.com/baidu-research/ba-dls-deepspeech
        """
        # extract strided windows
        truncate_size = (len(samples) - window_size) % stride_size
        samples = samples[:len(samples) - truncate_size]
        nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
        nstrides = (samples.strides[0], samples.strides[0] * stride_size)
        windows = np.lib.stride_tricks.as_strided(
            samples, shape=nshape, strides=nstrides)
        assert np.all(
            windows[:, 1] == samples[stride_size:(stride_size + window_size)])
        # window weighting, squared Fast Fourier Transform (fft), scaling
        weighting = np.hanning(window_size)[:, None]
        fft = np.fft.rfft(windows * weighting, axis=0)
        fft = np.absolute(fft)**2
        scale = np.sum(weighting**2) * sample_rate
        fft[1:-1, :] *= (2.0 / scale)
        fft[(0, -1), :] /= scale
        # prepare fft frequency list
        freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
        return fft, freqs
