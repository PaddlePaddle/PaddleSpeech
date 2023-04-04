# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# See the LICENSE file for licensing terms (BSD-style).
# Modified from https://github.com/webdataset/webdataset
#%%
import copy
import sys
from itertools import islice

from .paddle_utils import DataLoader
from .paddle_utils import IterableDataset
from .utils import PipelineStage


def add_length_method(obj):
    def length(self):
        return self.size

    Combined = type(
        obj.__class__.__name__ + "_Length",
        (obj.__class__, IterableDataset),
        {"__len__": length},
    )
    obj.__class__ = Combined
    return obj


class DataPipeline(IterableDataset, PipelineStage):
    """A pipeline starting with an IterableDataset and a series of filters."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pipeline = []
        self.length = -1
        self.repetitions = 1
        self.nsamples = -1
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, list):
                self.pipeline.extend(arg)
            else:
                self.pipeline.append(arg)

    def invoke(self, f, *args, **kwargs):
        """Apply a pipeline stage, possibly to the output of a previous stage."""
        if isinstance(f, PipelineStage):
            return f.run(*args, **kwargs)
        if isinstance(f, (IterableDataset, DataLoader)) and len(args) == 0:
            return iter(f)
        if isinstance(f, list):
            return iter(f)
        if callable(f):
            result = f(*args, **kwargs)
            return result
        raise ValueError(f"{f}: not a valid pipeline stage")

    def iterator1(self):
        """Create an iterator through one epoch in the pipeline."""
        source = self.invoke(self.pipeline[0])
        for step in self.pipeline[1:]:
            source = self.invoke(step, source)
        return source

    def iterator(self):
        """Create an iterator through the entire dataset, using the given number of repetitions."""
        for i in range(self.repetitions):
            for sample in self.iterator1():
                yield sample

    def __iter__(self):
        """Create an iterator through the pipeline, repeating and slicing as requested."""
        if self.repetitions != 1:
            if self.nsamples > 0:
                return islice(self.iterator(), self.nsamples)
            else:
                return self.iterator()
        else:
            return self.iterator()

    def stage(self, i):
        """Return pipeline stage i."""
        return self.pipeline[i]

    def append(self, f):
        """Append a pipeline stage (modifies the object)."""
        self.pipeline.append(f)
        return self

    def append_list(self, *args):
        for arg in args:
            self.pipeline.append(arg)
        return self

    def compose(self, *args):
        """Append a pipeline stage to a copy of the pipeline and returns the copy."""
        result = copy.copy(self)
        for arg in args:
            result.append(arg)
        return result

    def with_length(self, n):
        """Add a __len__ method returning the desired value.

        This does not change the actual number of samples in an epoch.
        PyTorch IterableDataset should not have a __len__ method.
        This is provided only as a workaround for some broken training environments
        that require a __len__ method.
        """
        self.size = n
        return add_length_method(self)

    def with_epoch(self, nsamples=-1, nbatches=-1):
        """Change the epoch to return the given number of samples/batches.

        The two arguments mean the same thing."""
        self.repetitions = sys.maxsize
        self.nsamples = max(nsamples, nbatches)
        return self

    def repeat(self, nepochs=-1, nbatches=-1):
        """Repeat iterating through the dataset for the given #epochs up to the given #samples."""
        if nepochs > 0:
            self.repetitions = nepochs
            self.nsamples = nbatches
        else:
            self.repetitions = sys.maxsize
            self.nsamples = nbatches
        return self
