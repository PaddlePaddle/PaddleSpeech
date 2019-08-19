"""Contain the resample augmentation model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils.augmentor.base import AugmentorBase


class ResampleAugmentor(AugmentorBase):
    """Augmentation model for resampling.

    See more info here:
    https://ccrma.stanford.edu/~jos/resample/index.html
    
    :param rng: Random generator object.
    :type rng: random.Random
    :param new_sample_rate: New sample rate in Hz.
    :type new_sample_rate: int
    """

    def __init__(self, rng, new_sample_rate):
        self._new_sample_rate = new_sample_rate
        self._rng = rng

    def transform_audio(self, audio_segment):
        """Resamples the input audio to a target sample rate.

        Note that this is an in-place transformation.

        :param audio: Audio segment to add effects to.
        :type audio: AudioSegment|SpeechSegment
        """
        audio_segment.resample(self._new_sample_rate)
