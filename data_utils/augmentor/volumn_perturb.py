from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from data_utils.augmentor.base import AugmentorBase


class VolumnPerturbAugmentor(AugmentorBase):
    def __init__(self, rng, min_gain_dBFS, max_gain_dBFS):
        self._min_gain_dBFS = min_gain_dBFS
        self._max_gain_dBFS = max_gain_dBFS
        self._rng = rng

    def transform_audio(self, audio_segment):
        gain = self._rng.uniform(min_gain_dBFS, max_gain_dBFS)
        audio_segment.apply_gain(gain)
