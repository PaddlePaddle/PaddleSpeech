from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import data_utils.utils as utils
from data_utils.audio import AudioSegment


class FeatureNormalizer(object):
    def __init__(self,
                 mean_std_filepath,
                 manifest_path=None,
                 featurize_func=None,
                 num_samples=500,
                 random_seed=0):
        if not mean_std_filepath:
            if not (manifest_path and featurize_func):
                raise ValueError("If mean_std_filepath is None, meanifest_path "
                                 "and featurize_func should not be None.")
            self._rng = random.Random(random_seed)
            self._compute_mean_std(manifest_path, featurize_func, num_samples)
        else:
            self._read_mean_std_from_file(mean_std_filepath)

    def apply(self, features, eps=1e-14):
        """Normalize features to be of zero mean and unit stddev."""
        return (features - self._mean) / (self._std + eps)

    def write_to_file(self, filepath):
        np.savez(filepath, mean=self._mean, std=self._std)

    def _read_mean_std_from_file(self, filepath):
        npzfile = np.load(filepath)
        self._mean = npzfile["mean"]
        self._std = npzfile["std"]

    def _compute_mean_std(self, manifest_path, featurize_func, num_samples):
        manifest = utils.read_manifest(manifest_path)
        sampled_manifest = self._rng.sample(manifest, num_samples)
        features = []
        for instance in sampled_manifest:
            features.append(
                featurize_func(
                    AudioSegment.from_file(instance["audio_filepath"])))
        features = np.hstack(features)
        self._mean = np.mean(features, axis=1).reshape([-1, 1])
        self._std = np.std(features, axis=1).reshape([-1, 1])
