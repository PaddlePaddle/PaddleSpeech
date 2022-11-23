"""PyTorch compatible DataLoaders

Essentially we extend PyTorch DataLoader by adding the ability to save the
data loading state, so that a checkpoint may be saved in the middle of an
epoch.

Example
-------
>>> import torch
>>> from speechbrain.utils.checkpoints import Checkpointer
>>> # An example "dataset" and its loader
>>> dataset = torch.randn(10, 1)
>>> dataloader = SaveableDataLoader(dataset, num_workers = 3)
>>> # Setup the checkpointer:
>>> tmpdir = getfixture('tmpdir')
>>> checkpointer = Checkpointer(tmpdir, {"dataloader": dataloader})
>>> # Iterate:
>>> for i, data_point in enumerate(dataloader):
...     # Here you would process the data:
...     rainfall_amount_prediction = data_point * 4.
...     # Now, imagine the experiment gets killed on the fifth batch:
...     if i == 4:
...         break
...     # Luckily, you had just saved a checkpoint:
...     if i == 3:
...         _ = checkpointer.save_checkpoint(end_of_epoch = False)
>>> # So when you restart the experiment:
>>> new_dataloader = SaveableDataLoader(dataset, num_workers = 3)
>>> new_checkpointer = Checkpointer(tmpdir, {"dataloader": new_dataloader})
>>> _ = new_checkpointer.recover_if_possible()
>>> # The dataloader fast-forwards to the position where we left off:
>>> assert next(iter(new_dataloader)) == dataset[4]

Authors:
  * Aku Rouhe 2020
"""
import collections
import torch
from paddlespeech.s2t.io.wav2vec2.data_utils import mod_default_collate
# from speechbrain.utils.data_utils import recursive_to
from paddlespeech.s2t.io.wav2vec2.data_utils import batch_pad_right
from paddle.io import DataLoader
import logging
import warnings
import functools
# from batch import PaddedBatch
from paddlespeech.s2t.io.wav2vec2.dataset import DynamicItemDataset
from paddlespeech.s2t.io.wav2vec2.sampler import ReproducibleRandomSampler
import paddle
PaddedData = collections.namedtuple("PaddedData", ["data", "lengths"])
import numpy

class Wav2vec2DataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=None,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False

        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
        if sampler is not None:
            self.batch_sampler.sampler = sampler
        # self.dataloader = DataLoader(
        #     dataset=dataset,
        #     batch_sampler=batch_sampler,
        #     collate_fn=collate_fn,
        #     num_workers=num_workers,)

    # def __len__(self):
    #     return len(self.dataloader)

    # def __iter__(self):
    #     return self.dataloader.__iter__()

    # def __call__(self):
    #     return self.__iter__()


def PaddedBatch(
        examples,
        padded_keys=None,
        device_prep_keys=None,
        padding_func=batch_pad_right,
        padding_kwargs={},
        nonpadded_stack=True,
    ):
    __length = len(examples)
    __keys = list(examples[0].keys())
    __padded_keys = []
    __device_prep_keys = []
    res = {}
    for key in __keys:
        values = [example[key] for example in examples]
        # Default convert usually does the right thing (numpy2torch etc.)
        # values = default_convert(values)
        if (padded_keys is not None and key in padded_keys) or (
            padded_keys is None and isinstance(values[0], numpy.ndarray)
        ):
            # Padding and PaddedData
            __padded_keys.append(key)

            padded = PaddedData(*padding_func(values, **padding_kwargs))
            res[key] = padded
        else:
            # Default PyTorch collate usually does the right thing
            # (convert lists of equal sized tensors to batch tensors, etc.)
            if nonpadded_stack:
                values = mod_default_collate(values)
            res[key] = values
        if (device_prep_keys is not None and key in device_prep_keys) or (
            device_prep_keys is None and isinstance(values[0], paddle.Tensor)
        ):
            __device_prep_keys.append(key) 
    return res

def make_dataloader(dataset, stage, **loader_kwargs):
    """Makes a basic DataLoader with SpeechBrain defaults.

    For DynamicItemDatasets (which return dicts), use
    PaddedBatch as the default collate_fn.

    Shuffling gets implemented by ReproducibleRandomSampler.

    If the Dataset is not an IterableDataset, the DataLoader
    is a SaveableDataLoader.

    If the Dataset is a webdataset.dataset.Composable, set default
    batch_size = None.

    Can also loop over the underlying dataloader continuously,
    and stop iterations at nominal epoch lengths.

    Arguments
    ---------
    dataset : Dataset
        The dataset to make a DataLoader for.
    looped_nominal_epoch : None, int
        If an integer is given, loop the underlying DataLoader infinitely and
        set a nominal epoch length in batches (or whatever the DataLoader
        yields).
    **loader_kwargs : dict
        Keyword args to DataLoader, see PyTorch DataLoader for
        options.

    Returns
    -------
    DataLoader
        If looped_nominal_epoch is None
    LoopedLoader
        If looped_nominal_epoch is not None
    """
    # PaddedBatch as default collation for DynamicItemDataset
    if "collate_fn" not in loader_kwargs and isinstance(
        dataset, DynamicItemDataset
    ):
        loader_kwargs["collate_fn"] = PaddedBatch
    # Reproducible random sampling
    if loader_kwargs.get("shuffle", False):
        if loader_kwargs.get("sampler") is not None:
            raise ValueError(
                "Cannot specify both shuffle=True and a "
                "sampler in loader_kwargs"
            )
        sampler = ReproducibleRandomSampler(dataset)
        loader_kwargs["sampler"] = sampler
        # Should delete shuffle because you can't set both Sampler and
        # shuffle
        # NOTE: the dict of loader options may get used elsewhere!
        # However, this del doesn't touch those because loader_kwargs comes
        # from a **kwargs dict.
        del loader_kwargs["shuffle"]
    # Create the loader
    dataloader = Wav2vec2DataLoader(dataset, **loader_kwargs)
    return dataloader


# import collections
# import torch
# from data_utils import mod_default_collate
# # from speechbrain.utils.data_utils import recursive_to
# from data_utils import batch_pad_right
# from torch.utils.data._utils.collate import default_convert
# # from torch.utils.data._utils.pin_memory import (
# #     pin_memory as recursive_pin_memory,
# # )
# import paddle

# PaddedData = collections.namedtuple("PaddedData", ["data", "lengths"])
