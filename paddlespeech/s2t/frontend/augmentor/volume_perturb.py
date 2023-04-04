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
"""Contains the volume perturb augmentation model."""
from paddlespeech.s2t.frontend.augmentor.base import AugmentorBase


class VolumePerturbAugmentor(AugmentorBase):
    """Augmentation model for adding random volume perturbation.
    
    This is used for multi-loudness training of PCEN. See

    https://arxiv.org/pdf/1607.05666v1.pdf

    for more details.

    :param rng: Random generator object.
    :type rng: random.Random
    :param min_gain_dBFS: Minimal gain in dBFS.
    :type min_gain_dBFS: float
    :param max_gain_dBFS: Maximal gain in dBFS.
    :type max_gain_dBFS: float
    """
    def __init__(self, rng, min_gain_dBFS, max_gain_dBFS):
        self._min_gain_dBFS = min_gain_dBFS
        self._max_gain_dBFS = max_gain_dBFS
        self._rng = rng

    def __call__(self, x, uttid=None, train=True):
        if not train:
            return x
        self.transform_audio(x)
        return x

    def transform_audio(self, audio_segment):
        """Change audio loadness.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        gain = self._rng.uniform(self._min_gain_dBFS, self._max_gain_dBFS)
        audio_segment.gain_db(gain)
