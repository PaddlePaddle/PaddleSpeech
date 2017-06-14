from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import base


class ResamplerAugmentor(base.AugmentorBase):
    """ Instantiates a resampler module.
    
    :param new_sample_rate: New sample rate in Hz
    :type new_sample_rate: func[int->scalar]
    :param rng: Random generator object.
    :type rng: random.Random
    """

    def __init__(self, rng, new_sample_rate):
        self.new_sample_rate = new_sample_rate
        self._rng = rng

    def transform_audio(self, audio_segment):
        """ Resamples the input audio to the target sample rate.

        Note that this is an in-place transformation.

        :param audio: input audio
        :type audio: SpeechDLSegment
        """
        new_sample_rate = self.new_sample_rate
        audio.resample(new_sample_rate)