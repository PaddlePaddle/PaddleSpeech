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
import numpy as np

from paddlespeech.s2t.frontend.augmentor.base import AugmentorBase


class SpeedPerturbAugmentor(AugmentorBase):
    """Augmentation model for adding speed perturbation."""
    def __init__(self,
                 rng,
                 min_speed_rate=0.9,
                 max_speed_rate=1.1,
                 num_rates=3):
        """speed perturbation.
        
        The speed perturbation in kaldi uses sox-speed instead of sox-tempo,
        and sox-speed just to resample the input,
        i.e pitch and tempo are changed both.

        "Why use speed option instead of tempo -s in SoX for speed perturbation"
        https://groups.google.com/forum/#!topic/kaldi-help/8OOG7eE4sZ8
    
        Sox speed:
        https://pysox.readthedocs.io/en/latest/api.html#sox.transform.Transformer
        
        See reference paper here:
        http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf
        
        Espnet:
        https://espnet.github.io/espnet/_modules/espnet/transform/perturb.html
        
        Nemo:
        https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/perturb.py#L92

        Args:
            rng (random.Random): Random generator object.
            min_speed_rate (float): Lower bound of new speed rate to sample and should
                not be smaller than 0.9.
            max_speed_rate (float): Upper bound of new speed rate to sample and should
                not be larger than 1.1.
            num_rates (int, optional): Number of discrete rates to allow. 
                Can be a positive or negative integer. Defaults to 3.
                If a positive integer greater than 0 is provided, the range of
                speed rates will be discretized into `num_rates` values.
                If a negative integer or 0 is provided, the full range of speed rates
                will be sampled uniformly.
                Note: If a positive integer is provided and the resultant discretized
                range of rates contains the value '1.0', then those samples with rate=1.0,
                will not be augmented at all and simply skipped. This is to unnecessary
                augmentation and increase computation time. Effective augmentation chance
                in such a case is = `prob * (num_rates - 1 / num_rates) * 100`% chance
                where `prob` is the global probability of a sample being augmented.

        Raises:
            ValueError: when speed_rate error
        """
        if min_speed_rate < 0.9:
            raise ValueError(
                "Sampling speed below 0.9 can cause unnatural effects")
        if max_speed_rate > 1.1:
            raise ValueError(
                "Sampling speed above 1.1 can cause unnatural effects")
        self._min_rate = min_speed_rate
        self._max_rate = max_speed_rate
        self._rng = rng
        self._num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(self._min_rate,
                                      self._max_rate,
                                      self._num_rates,
                                      endpoint=True)

    def __call__(self, x, uttid=None, train=True):
        if not train:
            return x
        self.transform_audio(x)
        return x

    def transform_audio(self, audio_segment):
        """Sample a new speed rate from the given range and
        changes the speed of the given audio clip.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegment|SpeechSegment
        """
        if self._num_rates < 0:
            speed_rate = self._rng.uniform(self._min_rate, self._max_rate)
        else:
            speed_rate = self._rng.choice(self._rates)

        # Skip perturbation in case of identity speed rate
        if speed_rate == 1.0:
            return

        audio_segment.change_speed(speed_rate)
