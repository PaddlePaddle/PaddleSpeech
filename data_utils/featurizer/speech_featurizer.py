from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils.featurizer.audio_featurizer import AudioFeaturizer
from data_utils.featurizer.text_featurizer import TextFeaturizer


class SpeechFeaturizer(object):
    def __init__(self,
                 vocab_filepath,
                 specgram_type='linear',
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_freq=None,
                 random_seed=0):
        self._audio_featurizer = AudioFeaturizer(
            specgram_type, stride_ms, window_ms, max_freq, random_seed)
        self._text_featurizer = TextFeaturizer(vocab_filepath)

    def featurize(self, speech_segment):
        audio_feature = self._audio_featurizer.featurize(speech_segment)
        text_ids = self._text_featurizer.text2ids(speech_segment.transcript)
        return audio_feature, text_ids

    @property
    def vocab_size(self):
        return self._text_featurizer.vocab_size

    @property
    def vocab_list(self):
        return self._text_featurizer.vocab_list
