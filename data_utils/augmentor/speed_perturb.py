"""Speed perturbation module for making ASR robust to different voice
types (high pitched, low pitched, etc)
Samples uniformly between speed_min and speed_max
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import base


class SpeedPerturbatioAugmentor(base.AugmentorBase):
    """ 
    Instantiates a speed perturbation module.

    See reference paper here:

    http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf

    :param speed_min: Lower bound on new rate to sample
    :type speed_min: func[int->scalar]
    :param speed_max: Upper bound on new rate to sample
    :type speed_max: func[int->scalar]
    """

    def __init__(self, rng, speed_min, speed_max):

        if (speed_min < 0.9):
            raise ValueError(
                "Sampling speed below 0.9 can cause unnatural effects")
        if (speed_min > 1.1):
            raise ValueError(
                "Sampling speed above 1.1 can cause unnatural effects")
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.rng = rng

    def transform_audio(self, audio_segment):
        """ 
        Samples a new speed rate from the given range and
        changes the speed of the given audio clip.

        Note that this is an in-place transformation.

        :param audio_segment: input audio
        :type audio_segment: SpeechDLSegment
        """
        read_size = 0
        speed_min = self.speed_min(iteration)
        speed_max = self.speed_max(iteration)
        sampled_speed = rng.uniform(speed_min, speed_max)
        audio = audio.change_speed(sampled_speed)
