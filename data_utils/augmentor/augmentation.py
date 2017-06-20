"""Contains the data augmentation pipeline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
from data_utils.augmentor.volume_perturb import VolumePerturbAugmentor
from data_utils.augmentor.speed_perturb import SpeedPerturbAugmentor
from data_utils.augmentor.resample import ResampleAugmentor
from data_utils.augmentor.online_bayesian_normalization import OnlineBayesianNormalizationAugmentor


class AugmentationPipeline(object):
    """Build a pre-processing pipeline with various augmentation models.Such a
    data augmentation pipeline is oftern leveraged to augment the training
    samples to make the model invariant to certain types of perturbations in the
    real world, improving model's generalization ability.

    The pipeline is built according the the augmentation configuration in json
    string, e.g.
    
    .. code-block::
        
        '[{"type": "volume",
           "params": {"min_gain_dBFS": -15,
                      "max_gain_dBFS": 15},
           "prob": 0.5},
          {"type": "speed",
           "params": {"min_speed_rate": 0.8,
                      "max_speed_rate": 1.2},
           "prob": 0.5}
         ]' 

    This augmentation configuration inserts two augmentation models
    into the pipeline, with one is VolumePerturbAugmentor and the other
    SpeedPerturbAugmentor. "prob" indicates the probability of the current
    augmentor to take effect.

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
            if self._rng.uniform(0., 1.) <= rate:
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
        if augmentor_type == "speed":
            return SpeedPerturbAugmentor(self._rng, **params)
        if augmentor_type == "resample":
            return ResampleAugmentor(self._rng, **params)
        if augmentor_type == "bayesian_normal":
            return OnlineBayesianNormalizationAugmentor(self._rng, **params)
        else:
            raise ValueError("Unknown augmentor type [%s]." % augmentor_type)
