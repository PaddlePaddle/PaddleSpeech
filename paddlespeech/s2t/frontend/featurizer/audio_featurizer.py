# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the audio featurizer class."""
import numpy as np
import paddle
import paddleaudio.compliance.kaldi as kaldi
from python_speech_features import delta
from python_speech_features import mfcc


class AudioFeaturizer():
    """Audio featurizer, for extracting features from audio contents of
    AudioSegment or SpeechSegment.

    Currently, it supports feature types of linear spectrogram and mfcc.

    :param spectrum_type: Specgram feature type. Options: 'linear'.
    :type spectrum_type: str
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :type stride_ms: float
    :param window_ms: Window size (in milliseconds) for generating frames.
    :type window_ms: float
    :param max_freq: When spectrum_type is 'linear', only FFT bins
                     corresponding to frequencies between [0, max_freq] are
                     returned; when spectrum_type is 'mfcc', max_feq is the
                     highest band edge of mel filters.
    :types max_freq: None|float
    :param target_sample_rate: Audio are resampled (if upsampling or
                               downsampling is allowed) to this before
                               extracting spectrogram features.
    :type target_sample_rate: float
    :param use_dB_normalization: Whether to normalize the audio to a certain
                                 decibels before extracting the features.
    :type use_dB_normalization: bool
    :param target_dB: Target audio decibels for normalization.
    :type target_dB: float
    """
    def __init__(self,
                 spectrum_type: str = 'linear',
                 feat_dim: int = None,
                 delta_delta: bool = False,
                 stride_ms=10.0,
                 window_ms=20.0,
                 n_fft=None,
                 max_freq=None,
                 target_sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20,
                 dither=1.0):
        self._spectrum_type = spectrum_type
        # mfcc and fbank using `feat_dim`
        self._feat_dim = feat_dim
        # mfcc and fbank using `delta-delta`
        self._delta_delta = delta_delta
        self._stride_ms = stride_ms
        self._window_ms = window_ms
        self._max_freq = max_freq
        self._target_sample_rate = target_sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self._fft_point = n_fft
        self._dither = dither

    def featurize(self,
                  audio_segment,
                  allow_downsampling=True,
                  allow_upsampling=True):
        """Extract audio features from AudioSegment or SpeechSegment.

        :param audio_segment: Audio/speech segment to extract features from.
        :type audio_segment: AudioSegment|SpeechSegment
        :param allow_downsampling: Whether to allow audio downsampling before
                                   featurizing.
        :type allow_downsampling: bool
        :param allow_upsampling: Whether to allow audio upsampling before
                                 featurizing.
        :type allow_upsampling: bool
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        :raises ValueError: If audio sample rate is not supported.
        """
        # upsampling or downsampling
        if ((audio_segment.sample_rate > self._target_sample_rate
             and allow_downsampling)
                or (audio_segment.sample_rate < self._target_sample_rate
                    and allow_upsampling)):
            audio_segment.resample(self._target_sample_rate)
        if audio_segment.sample_rate != self._target_sample_rate:
            raise ValueError("Audio sample rate is not supported. "
                             "Turn allow_downsampling or allow up_sampling on.")
        # decibel normalization
        if self._use_dB_normalization:
            audio_segment.normalize(target_db=self._target_dB)
        # extract spectrogram
        return self._compute_specgram(audio_segment)

    @property
    def stride_ms(self):
        return self._stride_ms

    @property
    def feature_size(self):
        """audio feature size"""
        feat_dim = 0
        if self._spectrum_type == 'linear':
            fft_point = self._window_ms if self._fft_point is None else self._fft_point
            feat_dim = int(fft_point * (self._target_sample_rate / 1000) / 2 +
                           1)
        elif self._spectrum_type == 'mfcc':
            # mfcc, delta, delta-delta
            feat_dim = int(self._feat_dim *
                           3) if self._delta_delta else int(self._feat_dim)
        elif self._spectrum_type == 'fbank':
            # fbank, delta, delta-delta
            feat_dim = int(self._feat_dim *
                           3) if self._delta_delta else int(self._feat_dim)
        else:
            raise ValueError("Unknown spectrum_type %s. "
                             "Supported values: linear." % self._spectrum_type)
        return feat_dim

    def _compute_specgram(self, audio_segment):
        """Extract various audio features."""
        sample_rate = audio_segment.sample_rate
        if self._spectrum_type == 'linear':
            samples = audio_segment.samples
            return self._compute_linear_specgram(samples,
                                                 sample_rate,
                                                 stride_ms=self._stride_ms,
                                                 window_ms=self._window_ms,
                                                 max_freq=self._max_freq)
        elif self._spectrum_type == 'mfcc':
            samples = audio_segment.to('int16')
            return self._compute_mfcc(samples,
                                      sample_rate,
                                      feat_dim=self._feat_dim,
                                      stride_ms=self._stride_ms,
                                      window_ms=self._window_ms,
                                      max_freq=self._max_freq,
                                      dither=self._dither,
                                      delta_delta=self._delta_delta)
        elif self._spectrum_type == 'fbank':
            samples = audio_segment.to('int16')
            return self._compute_fbank(samples,
                                       sample_rate,
                                       feat_dim=self._feat_dim,
                                       stride_ms=self._stride_ms,
                                       window_ms=self._window_ms,
                                       max_freq=self._max_freq,
                                       dither=self._dither,
                                       delta_delta=self._delta_delta)
        else:
            raise ValueError("Unknown spectrum_type %s. "
                             "Supported values: linear." % self._spectrum_type)

    def _specgram_real(self, samples, window_size, stride_size, sample_rate):
        """Compute the spectrogram for samples from a real signal."""
        # extract strided windows
        truncate_size = (len(samples) - window_size) % stride_size
        samples = samples[:len(samples) - truncate_size]
        nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
        nstrides = (samples.strides[0], samples.strides[0] * stride_size)
        windows = np.lib.stride_tricks.as_strided(samples,
                                                  shape=nshape,
                                                  strides=nstrides)
        assert np.all(windows[:, 1] == samples[stride_size:(stride_size +
                                                            window_size)])
        # window weighting, squared Fast Fourier Transform (fft), scaling
        weighting = np.hanning(window_size)[:, None]
        # https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
        fft = np.fft.rfft(windows * weighting, n=None, axis=0)
        fft = np.absolute(fft)
        fft = fft**2
        scale = np.sum(weighting**2) * sample_rate
        fft[1:-1, :] *= (2.0 / scale)
        fft[(0, -1), :] /= scale
        # prepare fft frequency list
        freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
        return fft, freqs

    def _compute_linear_specgram(self,
                                 samples,
                                 sample_rate,
                                 stride_ms=10.0,
                                 window_ms=20.0,
                                 max_freq=None,
                                 eps=1e-14):
        """Compute the linear spectrogram from FFT energy.

        Args:
            samples ([type]): [description]
            sample_rate ([type]): [description]
            stride_ms (float, optional): [description]. Defaults to 10.0.
            window_ms (float, optional): [description]. Defaults to 20.0.
            max_freq ([type], optional): [description]. Defaults to None.
            eps ([type], optional): [description]. Defaults to 1e-14.

        Raises:
            ValueError: [description]
            ValueError: [description]

        Returns:
            np.ndarray: log spectrogram, (time, freq)
        """
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             "sample rate.")
        if stride_ms > window_ms:
            raise ValueError("Stride size must not be greater than "
                             "window size.")
        stride_size = int(0.001 * sample_rate * stride_ms)
        window_size = int(0.001 * sample_rate * window_ms)
        specgram, freqs = self._specgram_real(samples,
                                              window_size=window_size,
                                              stride_size=stride_size,
                                              sample_rate=sample_rate)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
        # (freq, time)
        spec = np.log(specgram[:ind, :] + eps)
        return np.transpose(spec)

    def _concat_delta_delta(self, feat):
        """append delat, delta-delta feature.

        Args:
            feat (np.ndarray): (T, D)

        Returns:
            np.ndarray: feat with delta-delta, (T, 3*D)
        """
        # Deltas
        d_feat = delta(feat, 2)
        # Deltas-Deltas
        dd_feat = delta(feat, 2)
        # concat above three features
        concat_feat = np.concatenate((feat, d_feat, dd_feat), axis=1)
        return concat_feat

    def _compute_mfcc(self,
                      samples,
                      sample_rate,
                      feat_dim=13,
                      stride_ms=10.0,
                      window_ms=25.0,
                      max_freq=None,
                      dither=1.0,
                      delta_delta=True):
        """Compute mfcc from samples.

        Args:
            samples (np.ndarray, np.int16): the audio signal from which to compute features.
            sample_rate (float): the sample rate of the signal we are working with, in Hz.
            feat_dim (int): the number of cepstrum to return, default 13.
            stride_ms (float, optional): stride length in ms. Defaults to 10.0.
            window_ms (float, optional): window length in ms. Defaults to 25.0.
            max_freq ([type], optional): highest band edge of mel filters. In Hz, default is samplerate/2. Defaults to None.
            delta_delta (bool, optional): Whether with delta delta. Defaults to False.

        Raises:
            ValueError: max_freq > samplerate/2
            ValueError: stride_ms > window_ms

        Returns:
            np.ndarray: mfcc feature, (D, T).
        """
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             "sample rate.")
        if stride_ms > window_ms:
            raise ValueError("Stride size must not be greater than "
                             "window size.")
        # compute the 13 cepstral coefficients, and the first one is replaced
        # by log(frame energy), (T, D)
        mfcc_feat = mfcc(signal=samples,
                         samplerate=sample_rate,
                         winlen=0.001 * window_ms,
                         winstep=0.001 * stride_ms,
                         numcep=feat_dim,
                         nfilt=23,
                         nfft=512,
                         lowfreq=20,
                         highfreq=max_freq,
                         dither=dither,
                         remove_dc_offset=True,
                         preemph=0.97,
                         ceplifter=22,
                         useEnergy=True,
                         winfunc='povey')
        if delta_delta:
            mfcc_feat = self._concat_delta_delta(mfcc_feat)
        return mfcc_feat

    def _compute_fbank(self,
                       samples,
                       sample_rate,
                       feat_dim=40,
                       stride_ms=10.0,
                       window_ms=25.0,
                       max_freq=None,
                       dither=1.0,
                       delta_delta=False):
        """Compute logfbank from samples.
        
        Args:
            samples (np.ndarray, np.int16): the audio signal from which to compute features. Should be an N*1 array
            sample_rate (float): the sample rate of the signal we are working with, in Hz.
            feat_dim (int): the number of cepstrum to return, default 13.
            stride_ms (float, optional): stride length in ms. Defaults to 10.0.
            window_ms (float, optional): window length in ms. Defaults to 20.0.
            max_freq (float, optional): highest band edge of mel filters. In Hz, default is samplerate/2. Defaults to None.
            delta_delta (bool, optional): Whether with delta delta. Defaults to False.

        Raises:
            ValueError: max_freq > samplerate/2
            ValueError: stride_ms > window_ms

        Returns:
            np.ndarray: mfcc feature, (D, T).
        """
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             "sample rate.")
        if stride_ms > window_ms:
            raise ValueError("Stride size must not be greater than "
                             "window size.")
        # (T, D)
        waveform = paddle.to_tensor(np.expand_dims(samples, 0),
                                    dtype=paddle.float32)
        mat = kaldi.fbank(
            waveform,
            n_mels=feat_dim,
            frame_length=window_ms,  # default : 25
            frame_shift=stride_ms,  # default : 10
            dither=dither,
            energy_floor=0.0,
            sr=sample_rate)
        fbank_feat = np.squeeze(mat.numpy())
        if delta_delta:
            fbank_feat = self._concat_delta_delta(fbank_feat)
        return fbank_feat
