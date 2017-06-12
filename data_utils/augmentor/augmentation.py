from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
from data_utils.augmentor.volumn_perturb import VolumnPerturbAugmentor


class AugmentationPipeline(object):
    def __init__(self, augmentation_config, random_seed=0):
        self._rng = random.Random(random_seed)
        self._augmentors, self._rates = self._parse_pipeline_from(
            augmentation_config)

    def transform_audio(self, audio_segment):
        for augmentor, rate in zip(self._augmentors, self._rates):
            if self._rng.uniform(0., 1.) <= rate:
                augmentor.transform_audio(audio_segment)

    def _parse_pipeline_from(self, config_json):
        try:
            configs = json.loads(config_json)
        except Exception as e:
            raise ValueError("Augmentation config json format error: "
                             "%s" % str(e))
        augmentors = [
            self._get_augmentor(config["type"], config["params"])
            for config in configs
        ]
        rates = [config["rate"] for config in configs]
        return augmentors, rates

    def _get_augmentor(self, augmentor_type, params):
        if augmentor_type == "volumn":
            return VolumnPerturbAugmentor(self._rng, **params)
        else:
            raise ValueError("Unknown augmentor type [%s]." % augmentor_type)
