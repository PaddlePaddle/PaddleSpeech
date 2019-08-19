"""Contain the online bayesian normalization augmentation model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils.augmentor.base import AugmentorBase


class OnlineBayesianNormalizationAugmentor(AugmentorBase):
    """Augmentation model for adding online bayesian normalization.

    :param rng: Random generator object.
    :type rng: random.Random
    :param target_db: Target RMS value in decibels.
    :type target_db: float
    :param prior_db: Prior RMS estimate in decibels.
    :type prior_db: float
    :param prior_samples: Prior strength in number of samples.
    :type prior_samples: int
    :param startup_delay: Default 0.0s. If provided, this function will
                          accrue statistics for the first startup_delay 
                          seconds before applying online normalization.
    :type starup_delay: float.
    """

    def __init__(self,
                 rng,
                 target_db,
                 prior_db,
                 prior_samples,
                 startup_delay=0.0):
        self._target_db = target_db
        self._prior_db = prior_db
        self._prior_samples = prior_samples
        self._rng = rng
        self._startup_delay = startup_delay

    def transform_audio(self, audio_segment):
        """Normalizes the input audio using the online Bayesian approach.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegment|SpeechSegment
        """
        audio_segment.normalize_online_bayesian(self._target_db, self._prior_db,
                                                self._prior_samples,
                                                self._startup_delay)
