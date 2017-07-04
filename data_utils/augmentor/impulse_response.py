"""Contains the impulse response augmentation model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils.augmentor.base import AugmentorBase
from data_utils import utils
from data_utils.audio import AudioSegment


class ImpulseResponseAugmentor(AugmentorBase):
    """Augmentation model for adding impulse response effect.
    
    :param rng: Random generator object.
    :type rng: random.Random
    :param impulse_manifest: Manifest path for impulse audio data.
    :type impulse_manifest: basestring 
    """

    def __init__(self, rng, impulse_manifest):
        self._rng = rng
        self._manifest = utils.read_manifest(manifest_path=impulse_manifest)

    def transform_audio(self, audio_segment):
        """Add impulse response effect.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        noise_json = self._rng.sample(self._manifest, 1)[0]
        noise_segment = AudioSegment.from_file(noise_json['audio_filepath'])
        audio_segment.convolve(noise_segment, allow_resample=True)
