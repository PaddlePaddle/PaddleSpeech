# Copyright (c) 2023 speechbrain Authors. All Rights Reserved.
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#
# Modified from speechbrain 2023 (https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/dataio/dataloader.py)
"""Paddle compatible DataLoaders

Essentially we extend Paddle DataLoader by adding the ability to save the
data loading state, so that a checkpoint may be saved in the middle of an
epoch.

Authors:
  * Aku Rouhe 2020
"""
import collections
import functools
import logging
import warnings

import paddle
from paddle.io import DataLoader

from paddlespeech.s2t.io.speechbrain.data_utils import batch_pad_right
from paddlespeech.s2t.io.speechbrain.data_utils import mod_default_collate
from paddlespeech.s2t.io.speechbrain.dataset import DynamicItemDataset
from paddlespeech.s2t.io.speechbrain.sampler import ReproducibleRandomSampler
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


def PaddedBatch(
        examples,
        padded_keys=None,
        device_prep_keys=None,
        padding_func=batch_pad_right,
        padding_kwargs={},
        nonpadded_stack=True, ):
    __length = len(examples)
    __keys = list(examples[0].keys())
    __padded_keys = []
    __device_prep_keys = []
    res = {}
    for key in __keys:
        values = [example[key] for example in examples]
        # Default convert usually does the right thing (numpy2tensor etc.)
        # values = default_convert(values)
        if (padded_keys is not None and key in padded_keys) or (
                padded_keys is None and isinstance(values[0], numpy.ndarray)):
            # Padding and PaddedData
            __padded_keys.append(key)

            padded = PaddedData(*padding_func(values, **padding_kwargs))
            res[key] = padded
        else:
            # Default collate usually does the right thing
            # (convert lists of equal sized tensors to batch tensors, etc.)
            if nonpadded_stack:
                values = mod_default_collate(values)
            res[key] = values
        if (device_prep_keys is not None and key in device_prep_keys) or (
                device_prep_keys is None and
                isinstance(values[0], paddle.Tensor)):
            __device_prep_keys.append(key)
    return res


def make_dataloader(dataset, stage, **loader_kwargs):
    """Makes a basic DataLoader.

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
        Keyword args to DataLoader, see Paddle DataLoader for
        options.

    Returns
    -------
    DataLoader
        If looped_nominal_epoch is None
    LoopedLoader
        If looped_nominal_epoch is not None
    """
    # PaddedBatch as default collation for DynamicItemDataset
    if "collate_fn" not in loader_kwargs and isinstance(dataset,
                                                        DynamicItemDataset):
        loader_kwargs["collate_fn"] = PaddedBatch
    # Reproducible random sampling
    if loader_kwargs.get("shuffle", False):
        if loader_kwargs.get("sampler") is not None:
            raise ValueError("Cannot specify both shuffle=True and a "
                             "sampler in loader_kwargs")
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
