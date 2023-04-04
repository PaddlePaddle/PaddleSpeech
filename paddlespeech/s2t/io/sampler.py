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
import math

import numpy as np
from paddle import distributed as dist
from paddle.io import BatchSampler
from paddle.io import DistributedBatchSampler

from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = [
    "SortagradDistributedBatchSampler",
    "SortagradBatchSampler",
]


def _batch_shuffle(indices, batch_size, epoch, clipped=False):
    """Put similarly-sized instances into minibatches for better efficiency
    and make a batch-wise shuffle.

    1. Sort the audio clips by duration.
    2. Generate a random number `k`, k in [0, batch_size).
    3. Randomly shift `k` instances in order to create different batches
        for different epochs. Create minibatches.
    4. Shuffle the minibatches.

    :param indices: indexes. List of int.
    :type indices: list
    :param batch_size: Batch size. This size is also used for generate
                        a random number for batch shuffle.
    :type batch_size: int
    :param clipped: Whether to clip the heading (small shift) and trailing
                    (incomplete batch) instances.
    :type clipped: bool
    :return: Batch shuffled mainifest.
    :rtype: list
    """
    rng = np.random.RandomState(epoch)
    shift_len = rng.randint(0, batch_size - 1)
    batch_indices = list(zip(*[iter(indices[shift_len:])] * batch_size))
    rng.shuffle(batch_indices)
    batch_indices = [item for batch in batch_indices for item in batch]
    assert clipped is False
    if not clipped:
        res_len = len(indices) - shift_len - len(batch_indices)
        # when res_len is 0, will return whole list, len(List[-0:]) = len(List[:])
        if res_len != 0:
            batch_indices.extend(indices[-res_len:])
        batch_indices.extend(indices[0:shift_len])
        assert len(indices) == len(
            batch_indices
        ), f"_batch_shuffle: {len(indices)} : {len(batch_indices)} : {res_len} - {shift_len}"
    return batch_indices


class SortagradDistributedBatchSampler(DistributedBatchSampler):
    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False,
                 sortagrad=False,
                 shuffle_method="batch_shuffle"):
        """Sortagrad Sampler for multi gpus.

        Args:
            dataset (paddle.io.Dataset): 
            batch_size (int): batch size for one gpu
            num_replicas (int, optional): world size or numbers of gpus. Defaults to None.
            rank (int, optional): rank id. Defaults to None.
            shuffle (bool, optional): True for do shuffle, or else. Defaults to False.
            drop_last (bool, optional): whether drop last batch which is less than batch size. Defaults to False.
            sortagrad (bool, optional): True, do sortgrad in first epoch, then shuffle as usual; or else. Defaults to False.
            shuffle_method (str, optional): shuffle method, "instance_shuffle" or "batch_shuffle". Defaults to "batch_shuffle".
        """
        super().__init__(dataset, batch_size, num_replicas, rank, shuffle,
                         drop_last)
        self._sortagrad = sortagrad
        self._shuffle_method = shuffle_method

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # sort (by duration) or batch-wise shuffle the manifest
        if self.shuffle:
            if self.epoch == 0 and self._sortagrad:
                logger.info(
                    f'rank: {dist.get_rank()} dataset sortagrad! epoch {self.epoch}'
                )
            else:
                logger.info(
                    f'rank: {dist.get_rank()} dataset shuffle! epoch {self.epoch}'
                )
                if self._shuffle_method == "batch_shuffle":
                    # using `batch_size * nrank`, or will cause instability loss and nan or inf grad,
                    # since diff batch examlpe length in batches case instability loss in diff rank,
                    # e.g. rank0 maxlength 20, rank3 maxlength 1000
                    indices = _batch_shuffle(indices,
                                             self.batch_size * self.nranks,
                                             self.epoch,
                                             clipped=False)
                elif self._shuffle_method == "instance_shuffle":
                    np.random.RandomState(self.epoch).shuffle(indices)
                else:
                    raise ValueError("Unknown shuffle method %s." %
                                     self._shuffle_method)
        assert len(
            indices
        ) == self.total_size, f"batch shuffle examples error: {len(indices)} : {self.total_size}"

        # slice `self.batch_size` examples by rank id
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(
                indices[self.local_rank *
                        last_local_batch_size:(self.local_rank + 1) *
                        last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                logger.debug(
                    f"rank: {dist.get_rank()} batch index: {batch_indices} ")
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size


class SortagradBatchSampler(BatchSampler):
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=False,
                 drop_last=False,
                 sortagrad=False,
                 shuffle_method="batch_shuffle"):
        """Sortagrad Sampler for one gpu.

        Args:
            dataset (paddle.io.Dataset): 
            batch_size (int): batch size for one gpu
            shuffle (bool, optional): True for do shuffle, or else. Defaults to False.
            drop_last (bool, optional): whether drop last batch which is less than batch size. Defaults to False.
            sortagrad (bool, optional): True, do sortgrad in first epoch, then shuffle as usual; or else. Defaults to False.
            shuffle_method (str, optional): shuffle method, "instance_shuffle" or "batch_shuffle". Defaults to "batch_shuffle".
        """
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
            "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
            "drop_last should be a boolean number"

        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0))
        self.total_size = self.num_samples
        self._sortagrad = sortagrad
        self._shuffle_method = shuffle_method

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # sort (by duration) or batch-wise shuffle the manifest
        if self.shuffle:
            if self.epoch == 0 and self._sortagrad:
                logger.info(f'dataset sortagrad! epoch {self.epoch}')
            else:
                logger.info(f'dataset shuffle! epoch {self.epoch}')
                if self._shuffle_method == "batch_shuffle":
                    indices = _batch_shuffle(indices,
                                             self.batch_size,
                                             self.epoch,
                                             clipped=False)
                elif self._shuffle_method == "instance_shuffle":
                    np.random.RandomState(self.epoch).shuffle(indices)
                else:
                    raise ValueError("Unknown shuffle method %s." %
                                     self._shuffle_method)
        assert len(
            indices
        ) == self.total_size, f"batch shuffle examples error: {len(indices)} : {self.total_size}"

        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                logger.debug(
                    f"rank: {dist.get_rank()} batch index: {batch_indices} ")
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

        self.epoch += 1

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size
