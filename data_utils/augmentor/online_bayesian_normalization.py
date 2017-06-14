""" Online bayesian normalization
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import base


class OnlineBayesianNormalizationAugmentor(base.AugmentorBase):
    """ 
    Instantiates an online bayesian normalization module.
    :param target_db: Target RMS value in decibels
            :type target_db: func[int->scalar]
            :param prior_db: Prior RMS estimate in decibels
            :type prior_db: func[int->scalar]
            :param prior_samples: Prior strength in number of samples
            :type prior_samples: func[int->scalar]
            :param startup_delay: Start-up delay in seconds during
                which normalization statistics is accrued.
            :type starup_delay: func[int->scalar]
    """

    def __init__(self,
                 rng,
                 target_db,
                 prior_db,
                 prior_samples,
                 startup_delay=base.parse_parameter_from(0.0)):

        self.target_db = target_db
        self.prior_db = prior_db
        self.prior_samples = prior_samples
        self.startup_delay = startup_delay
        self.rng = rng

    def transform_audio(self, audio_segment):
        """
        Normalizes the input audio using the online Bayesian approach.

        :param audio_segment: input audio
        :type audio_segment: SpeechSegment
        :param iteration: current iteration
        :type iteration: int
        :param text: audio transcription
        :type text: basestring
        :param rng: RNG to use for augmentation
        :type rng: random.Random

        """
        read_size = 0
        target_db = self.target_db(iteration)
        prior_db = self.prior_db(iteration)
        prior_samples = self.prior_samples(iteration)
        startup_delay = self.startup_delay(iteration)
        audio.normalize_online_bayesian(
            target_db, prior_db, prior_samples, startup_delay=startup_delay)
