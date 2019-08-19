"""Contains the impulse response augmentation model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils.augmentor.base import AugmentorBase
from data_utils.utility import read_manifest
from data_utils.audio import AudioSegment


class ImpulseResponseAugmentor(AugmentorBase):
    """Augmentation model for adding impulse response effect.

    :param rng: Random generator object.
    :type rng: random.Random
    :param impulse_manifest_path: Manifest path for impulse audio data.
    :type impulse_manifest_path: basestring
    """

    def __init__(self, rng, impulse_manifest_path):
        self._rng = rng
        self._impulse_manifest = read_manifest(impulse_manifest_path)

    def transform_audio(self, audio_segment):
        """Add impulse response effect.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        impulse_json = self._rng.sample(self._impulse_manifest, 1)[0]
        impulse_segment = AudioSegment.from_file(impulse_json['audio_filepath'])
        audio_segment.convolve(impulse_segment, allow_resample=True)
