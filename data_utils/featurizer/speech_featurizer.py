"""Contains the speech featurizer class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils.featurizer.audio_featurizer import AudioFeaturizer
from data_utils.featurizer.text_featurizer import TextFeaturizer


class SpeechFeaturizer(object):
    """Speech featurizer, for extracting features from both audio and transcript
    contents of SpeechSegment.

    Currently, for audio parts, it supports feature types of linear
    spectrogram and mfcc; for transcript parts, it only supports char-level
    tokenizing and conversion into a list of token indices. Note that the
    token indexing order follows the given vocabulary file.

    :param vocab_filepath: Filepath to load vocabulary for token indices
                           conversion.
    :type specgram_type: basestring
    :param specgram_type: Specgram feature type. Options: 'linear', 'mfcc'.
    :type specgram_type: str
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :type stride_ms: float
    :param window_ms: Window size (in milliseconds) for generating frames.
    :type window_ms: float
    :param max_freq: When specgram_type is 'linear', only FFT bins
                     corresponding to frequencies between [0, max_freq] are
                     returned; when specgram_type is 'mfcc', max_freq is the
                     highest band edge of mel filters.
    :types max_freq: None|float
    :param target_sample_rate: Speech are resampled (if upsampling or
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
                 vocab_filepath,
                 specgram_type='linear',
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_freq=None,
                 target_sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20):
        self._audio_featurizer = AudioFeaturizer(
            specgram_type=specgram_type,
            stride_ms=stride_ms,
            window_ms=window_ms,
            max_freq=max_freq,
            target_sample_rate=target_sample_rate,
            use_dB_normalization=use_dB_normalization,
            target_dB=target_dB)
        self._text_featurizer = TextFeaturizer(vocab_filepath)

    def featurize(self, speech_segment, keep_transcription_text):
        """Extract features for speech segment.

        1. For audio parts, extract the audio features.
        2. For transcript parts, keep the original text or convert text string
           to a list of token indices in char-level.

        :param audio_segment: Speech segment to extract features from.
        :type audio_segment: SpeechSegment
        :return: A tuple of 1) spectrogram audio feature in 2darray, 2) list of
                 char-level token indices.
        :rtype: tuple
        """
        audio_feature = self._audio_featurizer.featurize(speech_segment)
        if keep_transcription_text:
            return audio_feature, speech_segment.transcript
        text_ids = self._text_featurizer.featurize(speech_segment.transcript)
        return audio_feature, text_ids

    @property
    def vocab_size(self):
        """Return the vocabulary size.

        :return: Vocabulary size.
        :rtype: int
        """
        return self._text_featurizer.vocab_size

    @property
    def vocab_list(self):
        """Return the vocabulary in list.

        :return: Vocabulary in list.
        :rtype: list
        """
        return self._text_featurizer.vocab_list
