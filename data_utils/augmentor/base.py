"""Contains the abstract base class for augmentation models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod


class AugmentorBase(object):
    """Abstract base class for augmentation model (augmentor) class.
    All augmentor classes should inherit from this class, and implement the
    following abstract methods.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def transform_audio(self, audio_segment):
        """Adds various effects to the input audio segment. Such effects
        will augment the training data to make the model invariant to certain
        types of perturbations in the real world, improving model's
        generalization ability.
        
        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        pass
