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
"""Contains feature normalizers."""
import json

import jsonlines
import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.io import Dataset

from paddlespeech.s2t.frontend.audio import AudioSegment
from paddlespeech.s2t.frontend.utility import load_cmvn
from paddlespeech.s2t.utils.log import Log

__all__ = ["FeatureNormalizer"]

logger = Log(__name__).getlog()


# https://github.com/PaddlePaddle/Paddle/pull/31481
class CollateFunc(object):
    def __init__(self, feature_func):
        self.feature_func = feature_func

    def __call__(self, batch):
        mean_stat = None
        var_stat = None
        number = 0
        for item in batch:
            audioseg = AudioSegment.from_file(item['feat'])
            feat = self.feature_func(audioseg)  #(T, D)

            sums = np.sum(feat, axis=0)
            if mean_stat is None:
                mean_stat = sums
            else:
                mean_stat += sums

            square_sums = np.sum(np.square(feat), axis=0)
            if var_stat is None:
                var_stat = square_sums
            else:
                var_stat += square_sums

            number += feat.shape[0]
        return number, mean_stat, var_stat


class AudioDataset(Dataset):
    def __init__(self, manifest_path, num_samples=-1, rng=None, random_seed=0):
        self._rng = rng if rng else np.random.RandomState(random_seed)

        with jsonlines.open(manifest_path, 'r') as reader:
            manifest = list(reader)

        if num_samples == -1:
            sampled_manifest = manifest
        else:
            sampled_manifest = self._rng.choice(
                manifest, num_samples, replace=False)
        self.items = sampled_manifest

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class FeatureNormalizer(object):
    """Feature normalizer. Normalize features to be of zero mean and unit
    stddev.

    if mean_std_filepath is provided (not None), the normalizer will directly
    initilize from the file. Otherwise, both manifest_path and featurize_func
    should be given for on-the-fly mean and stddev computing.

    :param mean_std_filepath: File containing the pre-computed mean and stddev.
    :type mean_std_filepath: None|str
    :param manifest_path: Manifest of instances for computing mean and stddev.
    :type meanifest_path: None|str
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
                 num_workers=0,
                 random_seed=0):
        if not mean_std_filepath:
            if not (manifest_path and featurize_func):
                raise ValueError("If mean_std_filepath is None, meanifest_path "
                                 "and featurize_func should not be None.")
            self._rng = np.random.RandomState(random_seed)
            self._compute_mean_std(manifest_path, featurize_func, num_samples,
                                   num_workers)
        else:
            self._read_mean_std_from_file(mean_std_filepath)

    def apply(self, features):
        """Normalize features to be of zero mean and unit stddev.

        :param features: Input features to be normalized.
        :type features: ndarray, shape (T, D)
        :param eps:  added to stddev to provide numerical stablibity.
        :type eps: float
        :return: Normalized features.
        :rtype: ndarray
        """
        return (features - self._mean) * self._istd

    def _read_mean_std_from_file(self, filepath, eps=1e-20):
        """Load mean and std from file."""
        filetype = filepath.split(".")[-1]
        mean, istd = load_cmvn(filepath, filetype=filetype)
        self._mean = np.expand_dims(mean, axis=0)
        self._istd = np.expand_dims(istd, axis=0)

    def write_to_file(self, filepath):
        """Write the mean and stddev to the file.

        :param filepath: File to write mean and stddev.
        :type filepath: str
        """
        with open(filepath, 'w') as fout:
            fout.write(json.dumps(self.cmvn_info))

    def _compute_mean_std(self,
                          manifest_path,
                          featurize_func,
                          num_samples,
                          num_workers,
                          batch_size=64,
                          eps=1e-20):
        """Compute mean and std from randomly sampled instances."""
        paddle.set_device('cpu')

        collate_func = CollateFunc(featurize_func)
        dataset = AudioDataset(manifest_path, num_samples, self._rng)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_func)

        with paddle.no_grad():
            all_mean_stat = None
            all_var_stat = None
            all_number = 0
            wav_number = 0
            for i, batch in enumerate(data_loader):
                number, mean_stat, var_stat = batch
                if i == 0:
                    all_mean_stat = mean_stat
                    all_var_stat = var_stat
                else:
                    all_mean_stat += mean_stat
                    all_var_stat += var_stat
                all_number += number
                wav_number += batch_size

                if wav_number % 1000 == 0:
                    logger.info(
                        f'process {wav_number} wavs,{all_number} frames.')

        self.cmvn_info = {
            'mean_stat': list(all_mean_stat.tolist()),
            'var_stat': list(all_var_stat.tolist()),
            'frame_num': all_number,
        }

        return self.cmvn_info
