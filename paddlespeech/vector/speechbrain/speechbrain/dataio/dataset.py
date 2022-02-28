"""Dataset examples for loading individual data points

Authors
  * Aku Rouhe 2020
  * Samuele Cornell 2020
"""

import copy
import bisect
import sys
import contextlib
from types import MethodType
from paddle.io import Dataset
from paddle.io import IterableDataset
from speechbrain.utils.data_pipeline import DataPipeline
from speechbrain.dataio.dataio import load_data_json, load_data_csv
# import logging

# logger = logging.getLogger(__name__)

from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

class SamplesDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

# the ConcatDataset is implemented with paddle
from typing import (
    List,
    Iterable,
    TypeVar)

class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.
    This class is useful to assemble different existing datasets.
    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

class DynamicItemDataset(Dataset):
    # 保存数据集的类，在原始的数据类的基础上进行了一些扩展
    # 扩展的目的是支持关键词key匹配
    # 原始的数据data中是一个字典形式
    def __init__(
        self, data, dynamic_items=[], output_keys=[],
    ):
        self.data = data

        # data_ids 中保存了整个数据集的索引ids
        self.data_ids = list(self.data.keys())

        # static_keys中保存了每个元素拥有的字段
        # 每个元素中一定要有id这个字段
        static_keys = list(self.data[self.data_ids[0]].keys())
        if "id" in static_keys:
            raise ValueError("The key 'id' is reserved for the data point id.")
        else:
            static_keys.append("id")
            # logger.info("static keys append the 'id' key to static_keys")
        # logger.info("now all static_keys field is {}".format(static_keys))
        self.pipeline = DataPipeline(static_keys, dynamic_items)
        self.set_output_keys(output_keys)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        # 获取一个音频数据之后就开始处理，最后返回的是处理之后的音频数据结果
        data_id = self.data_ids[index]
        data_point = self.data[data_id]
        return self.pipeline.compute_outputs({"id": data_id, **data_point})

    def add_dynamic_item(self, func, takes=None, provides=None):
        """Makes a new dynamic item available on the dataset.

        Two calling conventions. For DynamicItem objects, just use:
        add_dynamic_item(dynamic_item).
        But otherwise, should use:
        add_dynamic_item(func, takes, provides).

        See `speechbrain.utils.data_pipeline`.

        Arguments
        ---------
        func : callable, DynamicItem
            If a DynamicItem is given, adds that directly. Otherwise a
            DynamicItem is created, and this specifies the callable to use. If
            a generator function is given, then create a GeneratorDynamicItem.
            Otherwise creates a normal DynamicItem.
        takes : list, str
            List of keys. When func is called, each key is resolved to
            either an entry in the data or the output of another dynamic_item.
            The func is then called with these as positional arguments,
            in the same order as specified here.
            A single arg can be given directly.
        provides : str
            Unique key or keys that this provides.
        """
        self.pipeline.add_dynamic_item(func, takes, provides)

    def set_output_keys(self, keys):
        """Use this to change the output keys.

        These are the keys that are actually evaluated when a data point
        is fetched from the dataset.

        Arguments
        ---------
        keys : dict, list
            List of keys (str) to produce in output.

            If a dict is given; it is used to map internal keys to output keys.
            From the output_keys dict key:value pairs the key appears outside,
            and value is the internal key.
        """
        self.pipeline.set_output_keys(keys)

    @contextlib.contextmanager
    def output_keys_as(self, keys):
        """Context manager to temporarily set output keys.

        Example
        -------
        >>> dataset = DynamicItemDataset({"a":{"x":1,"y":2},"b":{"x":3,"y":4}},
        ...     output_keys = ["x"])
        >>> with dataset.output_keys_as(["y"]):
        ...     print(dataset[0])
        {'y': 2}
        >>> print(dataset[0])
        {'x': 1}

        NOTE
        ----
        Not thread-safe. While in this context manager, the output keys
        are affected for any call.
        """
        saved_output = self.pipeline.output_mapping
        self.pipeline.set_output_keys(keys)
        yield self
        self.pipeline.set_output_keys(saved_output)

    def filtered_sorted(
        self,
        key_min_value={},
        key_max_value={},
        key_test={},
        sort_key=None,
        reverse=False,
        select_n=None,
    ):
        """Get a filtered and/or sorted version of this, shares static data.

        The reason to implement these operations in the same method is that
        computing some dynamic items may be expensive, and this way the
        filtering and sorting steps don't need to compute the dynamic items
        twice.

        Arguments
        ---------
        key_min_value : dict
            Map from key (in data or in dynamic items) to limit, will only keep
            data_point if data_point[key] >= limit
        key_max_value : dict
            Map from key (in data or in dynamic items) to limit, will only keep
            data_point if data_point[key] <= limit
        key_test : dict
            Map from key (in data or in dynamic items) to func, will only keep
            data_point if bool(func(data_point[key])) == True
        sort_key : None, str
            If not None, sort by data_point[sort_key]. Default is ascending
            order.
        reverse : bool
            If True, sort in descending order.
        select_n : None, int
            If not None, only keep (at most) the first n filtered data_points.
            The possible sorting is applied, but only on the first n data
            points found. Meant for debugging.

        Returns
        -------
        FilteredSortedDynamicItemDataset
            Shares the static data, but has its own output keys and
            dynamic items (initially deep copied from this, so they have the
            same dynamic items available)

        NOTE
        ----
        Temporarily changes the output keys!
        """
        filtered_sorted_ids = self._filtered_sorted_ids(
            key_min_value, key_max_value, key_test, sort_key, reverse, select_n,
        )
        return FilteredSortedDynamicItemDataset(
            self, filtered_sorted_ids
        )  # NOTE: defined below

    def _filtered_sorted_ids(
        self,
        key_min_value={},
        key_max_value={},
        key_test={},
        sort_key=None,
        reverse=False,
        select_n=None,
    ):
        """Returns a list of data ids, fulfilling the sorting and filtering."""

        def combined_filter(computed):
            for key, limit in key_min_value.items():
                # NOTE: docstring promises >= so using that.
                # Mathematically could also use < for nicer syntax, but
                # maybe with some super special weird edge case some one can
                # depend on the >= operator
                if computed[key] >= limit:
                    continue
                return False
            for key, limit in key_max_value.items():
                if computed[key] <= limit:
                    continue
                return False
            for key, func in key_test.items():
                if bool(func(computed[key])):
                    continue
                return False
            return True

        temp_keys = (
            set(key_min_value.keys())
            | set(key_max_value.keys())
            | set(key_test.keys())
            | set([] if sort_key is None else [sort_key])
        )
        filtered_ids = []
        with self.output_keys_as(temp_keys):
            for i, data_id in enumerate(self.data_ids):
                if select_n is not None and len(filtered_ids) == select_n:
                    break
                data_point = self.data[data_id]
                data_point["id"] = data_id
                computed = self.pipeline.compute_outputs(data_point)
                if combined_filter(computed):
                    if sort_key is not None:
                        # Add (main sorting index, current index, data_id)
                        # So that we maintain current sorting and don't compare
                        # data_id values ever.
                        filtered_ids.append((computed[sort_key], i, data_id))
                    else:
                        filtered_ids.append(data_id)
        if sort_key is not None:
            filtered_sorted_ids = [
                tup[2] for tup in sorted(filtered_ids, reverse=reverse)
            ]
        else:
            filtered_sorted_ids = filtered_ids
        return filtered_sorted_ids

    @classmethod
    def from_json(
        cls, json_path, replacements={}, dynamic_items=[], output_keys=[]
    ):
        """Load a data prep JSON file and create a Dataset based on it."""
        data = load_data_json(json_path, replacements)
        return cls(data, dynamic_items, output_keys)

    @classmethod
    def from_csv(
        cls, csv_path, replacements={}, dynamic_items=[], output_keys=[]
    ):
        """Load a data prep CSV file and create a Dataset based on it."""
        data = load_data_csv(csv_path, replacements)
        return cls(data, dynamic_items, output_keys)

    @classmethod
    def from_arrow_dataset(
        cls, dataset, replacements={}, dynamic_items=[], output_keys=[]
    ):
        """Loading a prepared huggingface dataset"""
        # define an unbound method to generate puesdo keys
        def keys(self):
            return [i for i in range(dataset.__len__())]

        # bind this method to arrow dataset
        dataset.keys = MethodType(keys, dataset)
        return cls(dataset, dynamic_items, output_keys)


class FilteredSortedDynamicItemDataset(DynamicItemDataset):
    """Possibly filtered, possibly sorted DynamicItemDataset.

    Shares the static data (reference).
    Has its own dynamic_items and output_keys (deepcopy).
    """

    def __init__(self, from_dataset, data_ids):
        self.data = from_dataset.data
        self.data_ids = data_ids
        self.pipeline = copy.deepcopy(from_dataset.pipeline)

    @classmethod
    def from_json(
        cls, json_path, replacements={}, dynamic_items=None, output_keys=None
    ):
        raise TypeError("Cannot create SubsetDynamicItemDataset directly!")

    @classmethod
    def from_csv(
        cls, csv_path, replacements={}, dynamic_items=None, output_keys=None
    ):
        raise TypeError("Cannot create SubsetDynamicItemDataset directly!")


def add_dynamic_item(datasets, func, takes=None, provides=None):
    """Helper for adding the same item to multiple datasets."""
    for dataset in datasets:
        dataset.add_dynamic_item(func, takes, provides)


def set_output_keys(datasets, output_keys):
    """Helper for setting the same item to multiple datasets."""
    for dataset in datasets:
        dataset.set_output_keys(output_keys)
