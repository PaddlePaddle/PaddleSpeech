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
"""Contains the abstract base class for augmentation models."""
from abc import ABCMeta
from abc import abstractmethod


class AugmentorBase(object):
    """Abstract base class for augmentation model (augmentor) class.
    All augmentor classes should inherit from this class, and implement the
    following abstract methods.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def transform_audio(self, audio_segment):
        """Adds various effects to the input audio segment. Such effects
        will augment the training data to make the model invariant to certain
        types of perturbations in the real world, improving model's
        generalization ability.
        
        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        pass
