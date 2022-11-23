import copy
import contextlib
from types import MethodType
from paddle.io import Dataset
from paddlespeech.s2t.io.wav2vec2.data_pipeline import DataPipeline
from paddlespeech.s2t.io.wav2vec2.dataio import load_data_json, load_data_csv
import logging

logger = logging.getLogger(__name__)


class DynamicItemDataset(Dataset):
    """Dataset that reads, wrangles, and produces dicts.

    Each data point dict provides some items (by key), for example, a path to a
    wavefile with the key "wav_file". When a data point is fetched from this
    Dataset, more items are produced dynamically, based on pre-existing items
    and other dynamic created items. For example, a dynamic item could take the
    wavfile path and load the audio from the disk.

    The dynamic items can depend on other dynamic items: a suitable evaluation
    order is used automatically,  as long as there are no circular dependencies.

    A specified list of keys is collected in the output dict. These can be items
    in the original data or dynamic items. If some dynamic items are not
    requested, nor depended on by other requested items, they won't be computed.
    So for example if a user simply wants to iterate over the text, the
    time-consuming audio loading can be skipped.

    About the format:
    Takes a dict of dicts as the collection of data points to read/wrangle.
    The top level keys are data point IDs.
    Each data point (example) dict should have the same keys, corresponding to
    different items in that data point.

    Altogether the data collection could look like this:

    >>> data = {
    ...  "spk1utt1": {
    ...      "wav_file": "/path/to/spk1utt1.wav",
    ...      "text": "hello world",
    ...      "speaker": "spk1",
    ...      },
    ...  "spk1utt2": {
    ...      "wav_file": "/path/to/spk1utt2.wav",
    ...      "text": "how are you world",
    ...      "speaker": "spk1",
    ...      }
    ... }

    NOTE
    ----
        The top-level key, the data point id, is implicitly added as an item
        in the data point, with the key "id"

    Each dynamic item is configured by three things: a key, a func, and a list
    of argkeys. The key should be unique among all the items (dynamic or not) in
    each data point. The func is any callable, and it returns the dynamic item's
    value. The callable is called with the values of other items as specified
    by the argkeys list (as positional args, passed in the order specified by
    argkeys).

    The dynamic_items configuration could look like this:

    >>> import torch
    >>> dynamic_items = [
    ...     {"func": lambda l: torch.Tensor(l),
    ...     "takes": ["wav_loaded"],
    ...     "provides": "wav"},
    ...     {"func": lambda path: [ord(c)/100 for c in path],  # Fake "loading"
    ...     "takes": ["wav_file"],
    ...     "provides": "wav_loaded"},
    ...     {"func": lambda t: t.split(),
    ...     "takes": ["text"],
    ...     "provides": "words"}]

    With these, different views of the data can be loaded:

    >>> from speechbrain.dataio.dataloader import SaveableDataLoader
    >>> from speechbrain.dataio.batch import PaddedBatch
    >>> dataset = DynamicItemDataset(data, dynamic_items)
    >>> dataloader = SaveableDataLoader(dataset, collate_fn=PaddedBatch,
    ...     batch_size=2)
    >>> # First, create encoding for words:
    >>> dataset.set_output_keys(["words"])
    >>> encoding = {}
    >>> next_id = 1
    >>> for batch in dataloader:
    ...     for sent in batch.words:
    ...         for word in sent:
    ...             if word not in encoding:
    ...                 encoding[word] = next_id
    ...                 next_id += 1
    >>> # Next, add an encoded words_tensor dynamic item:
    >>> dataset.add_dynamic_item(
    ...     func = lambda ws: torch.tensor([encoding[w] for w in ws],
    ...             dtype=torch.long),
    ...     takes = ["words"],
    ...     provides = "words_encoded")
    >>> # Now we can get word and audio tensors:
    >>> dataset.set_output_keys(["id", "wav", "words_encoded"])
    >>> batch = next(iter(dataloader))
    >>> batch.id
    ['spk1utt1', 'spk1utt2']
    >>> batch.wav  # +ELLIPSIS
    PaddedData(data=tensor([[0.4700, 1.1200, ...
    >>> batch.words_encoded
    PaddedData(data=tensor([[1, 2, 0, 0],
            [3, 4, 5, 2]]), lengths=tensor([0.5000, 1.0000]))

    Output keys can also be a map:

    >>> dataset.set_output_keys({"id":"id", "signal": "wav", "words": "words_encoded"})
    >>> batch = next(iter(dataloader))
    >>> batch.words
    PaddedData(data=tensor([[1, 2, 0, 0],
            [3, 4, 5, 2]]), lengths=tensor([0.5000, 1.0000]))


    Arguments
    ---------
    data : dict
        Dictionary containing single data points (e.g. utterances).
    dynamic_items : list, optional
        Configuration for the dynamic items produced when fetching an example.
        List of DynamicItems or dicts with the format::
            func: <callable> # To be called
            takes: <list> # key or list of keys of args this takes
            provides: key # key or list of keys that this provides
    output_keys : dict, list, optional
        List of keys (either directly available in data or dynamic items)
        to include in the output dict when data points are fetched.

        If a dict is given; it is used to map internal keys to output keys.
        From the output_keys dict key:value pairs the key appears outside,
        and value is the internal key.
    """

    def __init__(
        self, data, dynamic_items=[], output_keys=[],
    ):
        self.data = data
        self.data_ids = list(self.data.keys())
        static_keys = list(self.data[self.data_ids[0]].keys())
        if "id" in static_keys:
            raise ValueError("The key 'id' is reserved for the data point id.")
        else:
            static_keys.append("id")
        self.pipeline = DataPipeline(static_keys, dynamic_items)
        self.set_output_keys(output_keys)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
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
            """Applies filter."""
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
            "Returns the keys."
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
