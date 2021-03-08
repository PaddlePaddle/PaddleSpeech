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
"""Contains the impulse response augmentation model."""

from deepspeech.frontend.augmentor.base import AugmentorBase
from deepspeech.frontend.utility import read_manifest
from deepspeech.frontend.audio import AudioSegment


class ImpulseResponseAugmentor(AugmentorBase):
    """Augmentation model for adding impulse response effect.

    :param rng: Random generator object.
    :type rng: random.Random
    :param impulse_manifest_path: Manifest path for impulse audio data.
    :type impulse_manifest_path: str
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
