"""Contains the volume perturb augmentation model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils.augmentor.base import AugmentorBase


class ShiftPerturbAugmentor(AugmentorBase):
    """Augmentation model for adding random shift perturbation.
    
    :param rng: Random generator object.
    :type rng: random.Random
    :param min_shift_ms: Minimal shift in milliseconds.
    :type min_shift_ms: float
    :param max_shift_ms: Maximal shift in milliseconds.
    :type max_shift_ms: float
    """

    def __init__(self, rng, min_shift_ms, max_shift_ms):
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms
        self._rng = rng

    def transform_audio(self, audio_segment):
        """Shift audio.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        shift_ms = self._rng.uniform(self._min_shift_ms, self._max_shift_ms)
        audio_segment.shift(shift_ms)
