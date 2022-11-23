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
"""Contain the resample augmentation model."""
from paddlespeech.s2t.frontend.augmentor.base import AugmentorBase


class ResampleAugmentor(AugmentorBase):
    """Augmentation model for resampling.

    See more info here:
    https://ccrma.stanford.edu/~jos/resample/index.html
    
    :param rng: Random generator object.
    :type rng: random.Random
    :param new_sample_rate: New sample rate in Hz.
    :type new_sample_rate: int
    """

    def __init__(self, rng, new_sample_rate):
        self._new_sample_rate = new_sample_rate
        self._rng = rng

    def __call__(self, x, uttid=None, train=True):
        if not train:
            return x
        self.transform_audio(x)
        return x

    def transform_audio(self, audio_segment):
        """Resamples the input audio to a target sample rate.

        Note that this is an in-place transformation.

        :param audio: Audio segment to add effects to.
        :type audio: AudioSegment|SpeechSegment
        """
        audio_segment.resample(self._new_sample_rate)
