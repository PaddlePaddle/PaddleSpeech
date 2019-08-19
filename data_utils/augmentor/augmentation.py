"""Contains the data augmentation pipeline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
from data_utils.augmentor.volume_perturb import VolumePerturbAugmentor
from data_utils.augmentor.shift_perturb import ShiftPerturbAugmentor
from data_utils.augmentor.speed_perturb import SpeedPerturbAugmentor
from data_utils.augmentor.noise_perturb import NoisePerturbAugmentor
from data_utils.augmentor.impulse_response import ImpulseResponseAugmentor
from data_utils.augmentor.resample import ResampleAugmentor
from data_utils.augmentor.online_bayesian_normalization import \
     OnlineBayesianNormalizationAugmentor


class AugmentationPipeline(object):
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

    def __init__(self, augmentation_config, random_seed=0):
        self._rng = random.Random(random_seed)
        self._augmentors, self._rates = self._parse_pipeline_from(
            augmentation_config)

    def transform_audio(self, audio_segment):
        """Run the pre-processing pipeline for data augmentation.

        Note that this is an in-place transformation.
        
        :param audio_segment: Audio segment to process.
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        for augmentor, rate in zip(self._augmentors, self._rates):
            if self._rng.uniform(0., 1.) < rate:
                augmentor.transform_audio(audio_segment)

    def _parse_pipeline_from(self, config_json):
        """Parse the config json to build a augmentation pipelien."""
        try:
            configs = json.loads(config_json)
            augmentors = [
                self._get_augmentor(config["type"], config["params"])
                for config in configs
            ]
            rates = [config["prob"] for config in configs]
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
        else:
            raise ValueError("Unknown augmentor type [%s]." % augmentor_type)
