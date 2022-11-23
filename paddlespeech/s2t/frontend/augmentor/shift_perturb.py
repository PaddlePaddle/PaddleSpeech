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

    def __call__(self, x, uttid=None, train=True):
        if not train:
            return x
        self.transform_audio(x)
        return x

    def transform_audio(self, audio_segment):
        """Shift audio.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        shift_ms = self._rng.uniform(self._min_shift_ms, self._max_shift_ms)
        audio_segment.shift(shift_ms)
