# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the noise perturb augmentation model."""
import jsonlines

from paddlespeech.s2t.frontend.audio import AudioSegment
from paddlespeech.s2t.frontend.augmentor.base import AugmentorBase


class NoisePerturbAugmentor(AugmentorBase):
    """Augmentation model for adding background noise.

    :param rng: Random generator object.
    :type rng: random.Random
    :param min_snr_dB: Minimal signal noise ratio, in decibels.
    :type min_snr_dB: float
    :param max_snr_dB: Maximal signal noise ratio, in decibels.
    :type max_snr_dB: float
    :param noise_manifest_path: Manifest path for noise audio data.
    :type noise_manifest_path: str
    """
    def __init__(self, rng, min_snr_dB, max_snr_dB, noise_manifest_path):
        self._min_snr_dB = min_snr_dB
        self._max_snr_dB = max_snr_dB
        self._rng = rng
        with jsonlines.open(noise_manifest_path, 'r') as reader:
            self._noise_manifest = list(reader)

    def __call__(self, x, uttid=None, train=True):
        if not train:
            return x
        self.transform_audio(x)
        return x

    def transform_audio(self, audio_segment):
        """Add background noise audio.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        noise_json = self._rng.choice(self._noise_manifest, 1, replace=False)[0]
        if noise_json['duration'] < audio_segment.duration:
            raise RuntimeError("The duration of sampled noise audio is smaller "
                               "than the audio segment to add effects to.")
        diff_duration = noise_json['duration'] - audio_segment.duration
        start = self._rng.uniform(0, diff_duration)
        end = start + audio_segment.duration
        noise_segment = AudioSegment.slice_from_file(
            noise_json['audio_filepath'], start=start, end=end)
        snr_dB = self._rng.uniform(self._min_snr_dB, self._max_snr_dB)
        audio_segment.add_noise(noise_segment,
                                snr_dB,
                                allow_downsampling=True,
                                rng=self._rng)
