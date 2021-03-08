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
"""Contain the speech perturbation augmentation model."""

from deepspeech.frontend.augmentor.base import AugmentorBase


class SpeedPerturbAugmentor(AugmentorBase):
    """Augmentation model for adding speed perturbation.

    See reference paper here:
    http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf

    :param rng: Random generator object.
    :type rng: random.Random
    :param min_speed_rate: Lower bound of new speed rate to sample and should
                           not be smaller than 0.9.
    :type min_speed_rate: float
    :param max_speed_rate: Upper bound of new speed rate to sample and should
                           not be larger than 1.1.
    :type max_speed_rate: float
    """

    def __init__(self, rng, min_speed_rate, max_speed_rate):
        if min_speed_rate < 0.9:
            raise ValueError(
                "Sampling speed below 0.9 can cause unnatural effects")
        if max_speed_rate > 1.1:
            raise ValueError(
                "Sampling speed above 1.1 can cause unnatural effects")
        self._min_speed_rate = min_speed_rate
        self._max_speed_rate = max_speed_rate
        self._rng = rng

    def transform_audio(self, audio_segment):
        """Sample a new speed rate from the given range and
        changes the speed of the given audio clip.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegment|SpeechSegment
        """
        sampled_speed = self._rng.uniform(self._min_speed_rate,
                                          self._max_speed_rate)
        audio_segment.change_speed(sampled_speed)
