"""Contains the speech featurizer class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils.featurizer.audio_featurizer import AudioFeaturizer
from data_utils.featurizer.text_featurizer import TextFeaturizer


class SpeechFeaturizer(object):
    """Speech featurizer, for extracting features from both audio and transcript
    contents of SpeechSegment.

    Currently, for audio parts, it only supports feature type of linear
    spectrogram; for transcript parts, it only supports char-level tokenizing
    and conversion into a list of token indices. Note that the token indexing
    order follows the given vocabulary file.

    :param vocab_filepath: Filepath to load vocabulary for token indices
                           conversion.
    :type specgram_type: basestring
    :param specgram_type: Specgram feature type. Options: 'linear'.
    :type specgram_type: str
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :type stride_ms: float
    :param window_ms: Window size (in milliseconds) for generating frames.
    :type window_ms: float
    :param max_freq: Used when specgram_type is 'linear', only FFT bins
                     corresponding to frequencies between [0, max_freq] are
                     returned.
    :types max_freq: None|float
    """

    def __init__(self,
                 vocab_filepath,
                 specgram_type='linear',
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_freq=None):
        self._audio_featurizer = AudioFeaturizer(specgram_type, stride_ms,
                                                 window_ms, max_freq)
        self._text_featurizer = TextFeaturizer(vocab_filepath)

    def featurize(self, speech_segment):
        """Extract features for speech segment.

        1. For audio parts, extract the audio features.
        2. For transcript parts, convert text string to a list of token indices
           in char-level.

        :param audio_segment: Speech segment to extract features from.
        :type audio_segment: SpeechSegment
        :return: A tuple of 1) spectrogram audio feature in 2darray, 2) list of
                 char-level token indices.
        :rtype: tuple
        """
        audio_feature = self._audio_featurizer.featurize(speech_segment)
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
