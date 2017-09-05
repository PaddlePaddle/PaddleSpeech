"""Contains the noise perturb augmentation model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils.augmentor.base import AugmentorBase
from data_utils.utility import read_manifest
from data_utils.audio import AudioSegment


class NoisePerturbAugmentor(AugmentorBase):
    """Augmentation model for adding background noise.

    :param rng: Random generator object.
    :type rng: random.Random
    :param min_snr_dB: Minimal signal noise ratio, in decibels.
    :type min_snr_dB: float
    :param max_snr_dB: Maximal signal noise ratio, in decibels.
    :type max_snr_dB: float
    :param noise_manifest_path: Manifest path for noise audio data.
    :type noise_manifest_path: basestring
    """

    def __init__(self, rng, min_snr_dB, max_snr_dB, noise_manifest_path):
        self._min_snr_dB = min_snr_dB
        self._max_snr_dB = max_snr_dB
        self._rng = rng
        self._noise_manifest = read_manifest(manifest_path=noise_manifest_path)

    def transform_audio(self, audio_segment):
        """Add background noise audio.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        noise_json = self._rng.sample(self._noise_manifest, 1)[0]
        if noise_json['duration'] < audio_segment.duration:
            raise RuntimeError("The duration of sampled noise audio is smaller "
                               "than the audio segment to add effects to.")
        diff_duration = noise_json['duration'] - audio_segment.duration
        start = self._rng.uniform(0, diff_duration)
        end = start + audio_segment.duration
        noise_segment = AudioSegment.slice_from_file(
            noise_json['audio_filepath'], start=start, end=end)
        snr_dB = self._rng.uniform(self._min_snr_dB, self._max_snr_dB)
        audio_segment.add_noise(
            noise_segment, snr_dB, allow_downsampling=True, rng=self._rng)
