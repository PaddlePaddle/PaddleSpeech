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
import random

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.io import Dataset

from deepspeech.frontend.audio import AudioSegment
from deepspeech.frontend.utility import load_cmvn
from deepspeech.frontend.utility import read_manifest

__all__ = ["FeatureNormalizer"]


class CollateFunc(object):
    ''' Collate function for AudioDataset
    '''

    def __init__(self):
        pass

    def __call__(self, batch):
        mean_stat = None
        var_stat = None
        number = 0
        for feat in batch:
            sums = np.sum(feat, axis=1)
            if mean_stat is None:
                mean_stat = sums
            else:
                mean_stat += sums

            square_sums = np.sum(np.square(feat), axis=1)
            if var_stat is None:
                var_stat = square_sums
            else:
                var_stat += square_sums

            number += feat.shape[1]
        return paddle.to_tensor(number), paddle.to_tensor(
            mean_stat), paddle.to_tensor(var_stat)
        #return number, mean_stat, var_stat


class AudioDataset(Dataset):
    def __init__(self, manifest_path, feature_func, num_samples=-1, rng=None):
        self.feature_func = feature_func
        self._rng = rng
        manifest = read_manifest(manifest_path)
        if num_samples == -1:
            sampled_manifest = manifest
        else:
            sampled_manifest = self._rng.sample(manifest, num_samples)
        self.items = sampled_manifest

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        key = self.items[idx]['feat']
        audioseg = AudioSegment.from_file(key)
        feat = self.feature_func(audioseg)  #(D, T)
        return feat


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
            self._rng = random.Random(random_seed)
            self._compute_mean_std(manifest_path, featurize_func, num_samples,
                                   num_workers)
        else:
            self._read_mean_std_from_file(mean_std_filepath)

    def apply(self, features):
        """Normalize features to be of zero mean and unit stddev.

        :param features: Input features to be normalized.
        :type features: ndarray, shape (D, T)
        :param eps:  added to stddev to provide numerical stablibity.
        :type eps: float
        :return: Normalized features.
        :rtype: ndarray
        """
        return (features - self._mean) * self._istd

    def _read_mean_std_from_file(self, filepath, eps=1e-20):
        """Load mean and std from file."""
        mean, istd = load_cmvn(filepath, filetype='json')
        self._mean = mean
        self._istd = istd

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
                          eps=1e-20):
        """Compute mean and std from randomly sampled instances."""
        # manifest = read_manifest(manifest_path)
        # if num_samples == -1:
        #     sampled_manifest = manifest
        # else:
        #     sampled_manifest = self._rng.sample(manifest, num_samples)
        # features = []
        # for instance in sampled_manifest:
        #     features.append(
        #         featurize_func(AudioSegment.from_file(instance["feat"])))
        # features = np.hstack(features)  #(D, T)
        # self._mean = np.mean(features, axis=1)  #(D,)
        # std = np.std(features, axis=1)  #(D,)
        # std = np.clip(std, eps, None)
        # self._istd = 1.0 / std

        collate_func = CollateFunc()

        dataset = AudioDataset(manifest_path, featurize_func, num_samples,
                               self._rng)

        batch_size = 20
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
            for batch in data_loader():
                number, mean_stat, var_stat = batch
                if all_mean_stat is None:
                    all_mean_stat = mean_stat
                    all_var_stat = var_stat
                else:
                    all_mean_stat += mean_stat
                    all_var_stat += var_stat
                all_number += number
                wav_number += batch_size

                if wav_number % 1000 == 0:
                    print('process {} wavs,{} frames'.format(wav_number,
                                                             int(all_number)))

        self.cmvn_info = {
            'mean_stat': list(all_mean_stat.tolist()),
            'var_stat': list(all_var_stat.tolist()),
            'frame_num': int(all_number),
        }

        return self.cmvn_info
