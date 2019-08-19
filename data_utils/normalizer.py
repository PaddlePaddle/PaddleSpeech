"""Contains feature normalizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from data_utils.utility import read_manifest
from data_utils.audio import AudioSegment


class FeatureNormalizer(object):
    """Feature normalizer. Normalize features to be of zero mean and unit
    stddev.

    if mean_std_filepath is provided (not None), the normalizer will directly
    initilize from the file. Otherwise, both manifest_path and featurize_func
    should be given for on-the-fly mean and stddev computing.

    :param mean_std_filepath: File containing the pre-computed mean and stddev.
    :type mean_std_filepath: None|basestring
    :param manifest_path: Manifest of instances for computing mean and stddev.
    :type meanifest_path: None|basestring
    :param featurize_func: Function to extract features. It should be callable
                           with ``featurize_func(audio_segment)``.
    :type featurize_func: None|callable
    :param num_samples: Number of random samples for computing mean and stddev.
    :type num_samples: int
    :param random_seed: Random seed for sampling instances.
    :type random_seed: int
    :raises ValueError: If both mean_std_filepath and manifest_path
                        (or both mean_std_filepath and featurize_func) are None.
    """

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
        """Normalize features to be of zero mean and unit stddev.

        :param features: Input features to be normalized.
        :type features: ndarray
        :param eps:  added to stddev to provide numerical stablibity.
        :type eps: float
        :return: Normalized features.
        :rtype: ndarray
        """
        return (features - self._mean) / (self._std + eps)

    def write_to_file(self, filepath):
        """Write the mean and stddev to the file.

        :param filepath: File to write mean and stddev.
        :type filepath: basestring
        """
        np.savez(filepath, mean=self._mean, std=self._std)

    def _read_mean_std_from_file(self, filepath):
        """Load mean and std from file."""
        npzfile = np.load(filepath)
        self._mean = npzfile["mean"]
        self._std = npzfile["std"]

    def _compute_mean_std(self, manifest_path, featurize_func, num_samples):
        """Compute mean and std from randomly sampled instances."""
        manifest = read_manifest(manifest_path)
        sampled_manifest = self._rng.sample(manifest, num_samples)
        features = []
        for instance in sampled_manifest:
            features.append(
                featurize_func(
                    AudioSegment.from_file(instance["audio_filepath"])))
        features = np.hstack(features)
        self._mean = np.mean(features, axis=1).reshape([-1, 1])
        self._std = np.std(features, axis=1).reshape([-1, 1])
