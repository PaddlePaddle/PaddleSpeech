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
from collections.abc import Sequence
from inspect import signature

import numpy as np

from deepspeech.utils.dynamic_import import dynamic_import
from deepspeech.utils.log import Log

__all__ = ["AugmentationPipeline"]

logger = Log(__name__).getlog()

import_alias = dict(
    volume="deepspeech.frontend.augmentor.impulse_response:VolumePerturbAugmentor",
    shift="deepspeech.frontend.augmentor.shift_perturb:ShiftPerturbAugmentor",
    speed="deepspeech.frontend.augmentor.speed_perturb:SpeedPerturbAugmentor",
    resample="deepspeech.frontend.augmentor.resample:ResampleAugmentor",
    bayesian_normal="deepspeech.frontend.augmentor.online_bayesian_normalization:OnlineBayesianNormalizationAugmentor",
    noise="deepspeech.frontend.augmentor.noise_perturb:NoisePerturbAugmentor",
    impulse="deepspeech.frontend.augmentor.impulse_response:ImpulseResponseAugmentor",
    specaug="deepspeech.frontend.augmentor.spec_augment:SpecAugmentor", )


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

    Params:
        augmentation_config(str): Augmentation configuration in json string.
        random_seed(int): Random seed.
        train(bool): whether is train mode.
    
    Raises:
        ValueError: If the augmentation json config is in incorrect format".
    """

    def __init__(self, augmentation_config: str, random_seed: int=0):
        self._rng = np.random.RandomState(random_seed)
        self._spec_types = ('specaug')

        if augmentation_config is None:
            self.conf = {}
        else:
            self.conf = json.loads(augmentation_config)

        self._augmentors, self._rates = self._parse_pipeline_from('all')
        self._audio_augmentors, self._audio_rates = self._parse_pipeline_from(
            'audio')
        self._spec_augmentors, self._spec_rates = self._parse_pipeline_from(
            'feature')

    def __call__(self, xs, uttid_list=None, **kwargs):
        if not isinstance(xs, Sequence):
            is_batch = False
            xs = [xs]
        else:
            is_batch = True

        if isinstance(uttid_list, str):
            uttid_list = [uttid_list for _ in range(len(xs))]

        if self.conf.get("mode", "sequential") == "sequential":
            for idx, (func, rate) in enumerate(
                    zip(self._augmentors, self._rates), 0):
                if self._rng.uniform(0., 1.) >= rate:
                    continue

                # Derive only the args which the func has
                try:
                    param = signature(func).parameters
                except ValueError:
                    # Some function, e.g. built-in function, are failed
                    param = {}
                _kwargs = {k: v for k, v in kwargs.items() if k in param}

                try:
                    if uttid_list is not None and "uttid" in param:
                        xs = [
                            func(x, u, **_kwargs)
                            for x, u in zip(xs, uttid_list)
                        ]
                    else:
                        xs = [func(x, **_kwargs) for x in xs]
                except Exception:
                    logger.fatal("Catch a exception from {}th func: {}".format(
                        idx, func))
                    raise
        else:
            raise NotImplementedError(
                "Not supporting mode={}".format(self.conf["mode"]))

        if is_batch:
            return xs
        else:
            return xs[0]

    def transform_audio(self, audio_segment):
        """Run the pre-processing pipeline for data augmentation.

        Note that this is an in-place transformation.
        
        :param audio_segment: Audio segment to process.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        for augmentor, rate in zip(self._audio_augmentors, self._audio_rates):
            if self._rng.uniform(0., 1.) < rate:
                augmentor.transform_audio(audio_segment)

    def transform_feature(self, spec_segment):
        """spectrogram augmentation.
         
        Args:
            spec_segment (np.ndarray): audio feature, (D, T).
        """
        for augmentor, rate in zip(self._spec_augmentors, self._spec_rates):
            if self._rng.uniform(0., 1.) < rate:
                spec_segment = augmentor.transform_feature(spec_segment)
        return spec_segment

    def _parse_pipeline_from(self, aug_type='all'):
        """Parse the config json to build a augmentation pipelien."""
        assert aug_type in ('audio', 'feature', 'all'), aug_type
        audio_confs = []
        feature_confs = []
        all_confs = []
        for config in self.conf:
            all_confs.append(config)
            if config["type"] in self._spec_types:
                feature_confs.append(config)
            else:
                audio_confs.append(config)

        if aug_type == 'audio':
            aug_confs = audio_confs
        elif aug_type == 'feature':
            aug_confs = feature_confs
        else:
            aug_confs = all_confs

        augmentors = [
            self._get_augmentor(config["type"], config["params"])
            for config in aug_confs
        ]
        rates = [config["prob"] for config in aug_confs]
        return augmentors, rates

    def _get_augmentor(self, augmentor_type, params):
        """Return an augmentation model by the type name, and pass in params."""
        class_obj = dynamic_import(augmentor_type, import_alias)
        try:
            obj = class_obj(self._rng, **params)
        except Exception:
            raise ValueError("Unknown augmentor type [%s]." % augmentor_type)
        return obj
