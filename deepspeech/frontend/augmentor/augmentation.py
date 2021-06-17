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
"""Contains the data augmentation pipeline."""
import json

import numpy as np

from deepspeech.frontend.augmentor.impulse_response import ImpulseResponseAugmentor
from deepspeech.frontend.augmentor.noise_perturb import NoisePerturbAugmentor
from deepspeech.frontend.augmentor.online_bayesian_normalization import \
    OnlineBayesianNormalizationAugmentor
from deepspeech.frontend.augmentor.resample import ResampleAugmentor
from deepspeech.frontend.augmentor.shift_perturb import ShiftPerturbAugmentor
from deepspeech.frontend.augmentor.spec_augment import SpecAugmentor
from deepspeech.frontend.augmentor.speed_perturb import SpeedPerturbAugmentor
from deepspeech.frontend.augmentor.volume_perturb import VolumePerturbAugmentor


class AugmentationPipeline():
    """Build a pre-processing pipeline with various augmentation models.Such a
    data augmentation pipeline is oftern leveraged to augment the training
    samples to make the model invariant to certain types of perturbations in the
    real world, improving model's generalization ability.

    The pipeline is built according the the augmentation configuration in json
    string, e.g.
    
    .. code-block::

        [ {
                "type": "noise",
                "params": {"min_snr_dB": 10,
                           "max_snr_dB": 20,
                           "noise_manifest_path": "datasets/manifest.noise"},
                "prob": 0.0
            },
            {
                "type": "speed",
                "params": {"min_speed_rate": 0.9,
                           "max_speed_rate": 1.1},
                "prob": 1.0
            },
            {
                "type": "shift",
                "params": {"min_shift_ms": -5,
                           "max_shift_ms": 5},
                "prob": 1.0
            },
            {
                "type": "volume",
                "params": {"min_gain_dBFS": -10,
                           "max_gain_dBFS": 10},
                "prob": 0.0
            },
            {
                "type": "bayesian_normal",
                "params": {"target_db": -20,
                           "prior_db": -20,
                           "prior_samples": 100},
                "prob": 0.0
            }
        ]
        
    This augmentation configuration inserts two augmentation models
    into the pipeline, with one is VolumePerturbAugmentor and the other
    SpeedPerturbAugmentor. "prob" indicates the probability of the current
    augmentor to take effect. If "prob" is zero, the augmentor does not take
    effect.

    :param augmentation_config: Augmentation configuration in json string.
    :type augmentation_config: str
    :param random_seed: Random seed.
    :type random_seed: int
    :raises ValueError: If the augmentation json config is in incorrect format".
    """

    def __init__(self, augmentation_config: str, random_seed=0):
        self._rng = np.random.RandomState(random_seed)
        self._spec_types = ('specaug')
        self._augmentors, self._rates = self._parse_pipeline_from(
            augmentation_config, 'audio')
        self._spec_augmentors, self._spec_rates = self._parse_pipeline_from(
            augmentation_config, 'feature')

    def transform_audio(self, audio_segment, single=True):
        """Run the pre-processing pipeline for data augmentation.

        Note that this is an in-place transformation.
        
        :param audio_segment: Audio segment to process.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        for augmentor, rate in zip(self._augmentors, self._rates):
            if self._rng.uniform(0., 1.) < rate:
                augmentor.transform_audio(audio_segment, single)

    def transform_feature(self, spec_segment, single=True):
        """spectrogram augmentation.
         
        Args:
            spec_segment (np.ndarray): audio feature, (D, T).
        """
        for augmentor, rate in zip(self._spec_augmentors, self._spec_rates):
            if self._rng.uniform(0., 1.) < rate:
                spec_segment = augmentor.transform_feature(spec_segment, single)
        return spec_segment

    def _parse_pipeline_from(self, config_json, aug_type='audio'):
        """Parse the config json to build a augmentation pipelien."""
        assert aug_type in ('audio', 'feature'), aug_type
        try:
            configs = json.loads(config_json)
            audio_confs = []
            feature_confs = []
            for config in configs:
                if config["type"] in self._spec_types:
                    feature_confs.append(config)
                else:
                    audio_confs.append(config)

            if aug_type == 'audio':
                aug_confs = audio_confs
            elif aug_type == 'feature':
                aug_confs = feature_confs

            augmentors = [
                self._get_augmentor(config["type"], config["params"])
                for config in aug_confs
            ]
            rates = [config["prob"] for config in aug_confs]

        except Exception as e:
            raise ValueError("Failed to parse the augmentation config json: "
                             "%s" % str(e))
        return augmentors, rates

    def _get_augmentor(self, augmentor_type, params):
        """Return an augmentation model by the type name, and pass in params."""
        if augmentor_type == "volume":
            return VolumePerturbAugmentor(self._rng, **params)
        elif augmentor_type == "shift":
            return ShiftPerturbAugmentor(self._rng, **params)
        elif augmentor_type == "speed":
            return SpeedPerturbAugmentor(self._rng, **params)
        elif augmentor_type == "resample":
            return ResampleAugmentor(self._rng, **params)
        elif augmentor_type == "bayesian_normal":
            return OnlineBayesianNormalizationAugmentor(self._rng, **params)
        elif augmentor_type == "noise":
            return NoisePerturbAugmentor(self._rng, **params)
        elif augmentor_type == "impulse":
            return ImpulseResponseAugmentor(self._rng, **params)
        elif augmentor_type == "specaug":
            return SpecAugmentor(self._rng, **params)
        else:
            raise ValueError("Unknown augmentor type [%s]." % augmentor_type)
