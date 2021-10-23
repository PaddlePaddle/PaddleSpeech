# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import six
from paddle.io import Dataset

__all__ = [
    "split",
    "TransformDataset",
    "CacheDataset",
    "TupleDataset",
    "DictDataset",
    "SliceDataset",
    "SubsetDataset",
    "FilterDataset",
    "ChainDataset",
]


def split(dataset, first_size):
    """A utility function to split a dataset into two datasets."""
    first = SliceDataset(dataset, 0, first_size)
    second = SliceDataset(dataset, first_size, len(dataset))
    return first, second


class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        """Dataset which is transformed from another with a transform.

        Args:
            dataset (Dataset): the base dataset.
            transform (callable): the transform which takes an example of the base dataset as parameter and return a new example.
        """
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, i):
        in_data = self._dataset[i]
        return self._transform(in_data)


class CacheDataset(Dataset):
    def __init__(self, dataset):
        """A lazy cache of the base dataset.

        Args:
            dataset (Dataset): the base dataset to cache.
        """
        self._dataset = dataset
        self._cache = dict()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, i):
        if i not in self._cache:
            self._cache[i] = self._dataset[i]
        return self._cache[i]


class TupleDataset(Dataset):
    def __init__(self, *datasets):
        """A compound dataset made from several datasets of the same length. An example of the `TupleDataset` is a tuple of examples from the constituent datasets.

        Args:
            datasets: tuple[Dataset], the constituent datasets.
        """
        if not datasets:
            raise ValueError("no datasets are given")
        length = len(datasets[0])
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError("all the datasets should have the same length."
                                 "dataset {} has a different length".format(i))
        self._datasets = datasets
        self._length = length

    def __getitem__(self, index):
        # SOA
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, slice):
            length = len(batches[0])
            # AOS
            return [
                tuple([batch[i] for batch in batches])
                for i in six.moves.range(length)
            ]
        else:
            return tuple(batches)

    def __len__(self):
        return self._length


class DictDataset(Dataset):
    def __init__(self, **datasets):
        """
        A compound dataset made from several datasets of the same length. An 
        example of the `DictDataset` is a dict of examples from the constituent 
        datasets.

        WARNING: paddle does not have a good support for DictDataset, because
        every batch yield from a DataLoader is a list, but it cannot be a dict.
        So you have to provide a collate function because you cannot use the
        default one.

        Args:
            datasets: Dict[Dataset], the constituent datasets.
        """
        if not datasets:
            raise ValueError("no datasets are given")
        length = None
        for key, dataset in six.iteritems(datasets):
            if length is None:
                length = len(dataset)
            elif len(dataset) != length:
                raise ValueError(
                    "all the datasets should have the same length."
                    "dataset {} has a different length".format(key))
        self._datasets = datasets
        self._length = length

    def __getitem__(self, index):
        batches = {
            key: dataset[index]
            for key, dataset in six.iteritems(self._datasets)
        }
        if isinstance(index, slice):
            length = len(six.next(six.itervalues(batches)))
            return [{key: batch[i]
                     for key, batch in six.iteritems(batches)}
                    for i in six.moves.range(length)]
        else:
            return batches

    def __len__(self):
        return self._length


class SliceDataset(Dataset):
    def __init__(self, dataset, start, finish, order=None):
        """A Dataset which is a slice of the base dataset.

        Args:
            dataset (Dataset): the base dataset.
            start (int): the start of the slice.
            finish (int): the end of the slice, not inclusive.
            order (List[int], optional): the order, it is a permutation of the valid example ids of the base dataset. If `order` is provided, the slice is taken in `order`. Defaults to None.
        """
        if start < 0 or finish > len(dataset):
            raise ValueError("subset overruns the dataset.")
        self._dataset = dataset
        self._start = start
        self._finish = finish
        self._size = finish - start

        if order is not None and len(order) != len(dataset):
            raise ValueError(
                "order should have the same length as the dataset"
                "len(order) = {} which does not euqals len(dataset) = {} ".
                format(len(order), len(dataset)))
        self._order = order

    def __len__(self):
        return self._size

    def __getitem__(self, i):
        if i >= 0:
            if i >= self._size:
                raise IndexError('dataset index out of range')
            index = self._start + i
        else:
            if i < -self._size:
                raise IndexError('dataset index out of range')
            index = self._finish + i

        if self._order is not None:
            index = self._order[index]
        return self._dataset[index]


class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        """A Dataset which is a subset of the base dataset.

        Args:
            dataset (Dataset): the base dataset.
            indices (Iterable[int]): the indices of the examples to pick.
        """
        self._dataset = dataset
        if len(indices) > len(dataset):
            raise ValueError("subset's size larger that dataset's size!")
        self._indices = indices
        self._size = len(indices)

    def __len__(self):
        return self._size

    def __getitem__(self, i):
        index = self._indices[i]
        return self._dataset[index]


class FilterDataset(Dataset):
    def __init__(self, dataset, filter_fn):
        """A filtered dataset.

        Args:
            dataset (Dataset): the base dataset.
            filter_fn (callable): a callable which takes an example of the base dataset and return a boolean.
        """
        self._dataset = dataset
        self._indices = [
            i for i in range(len(dataset)) if filter_fn(dataset[i])
        ]
        self._size = len(self._indices)

    def __len__(self):
        return self._size

    def __getitem__(self, i):
        index = self._indices[i]
        return self._dataset[index]


class ChainDataset(Dataset):
    def __init__(self, *datasets):
        """A concatenation of the several datasets which the same structure.

        Args:
            datasets (Iterable[Dataset]): datasets to concat.
        """
        self._datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, i):
        if i < 0:
            raise IndexError("ChainDataset doesnot support negative indexing.")

        for dataset in self._datasets:
            if i < len(dataset):
                return dataset[i]
            i -= len(dataset)

        raise IndexError("dataset index out of range")
