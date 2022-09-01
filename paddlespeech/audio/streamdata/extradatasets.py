#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
# Modified from https://github.com/webdataset/webdataset
#
"""Train PyTorch models directly from POSIX tar archive.

Code works locally or over HTTP connections.
"""
from . import utils
from .paddle_utils import IterableDataset
from .utils import PipelineStage


class MockDataset(IterableDataset):
    """MockDataset.

    A mock dataset for performance testing and unit testing.
    """

    def __init__(self, sample, length):
        """Create a mock dataset instance.

        :param sample: the sample to be returned repeatedly
        :param length: the length of the mock dataset
        """
        self.sample = sample
        self.length = length

    def __iter__(self):
        """Return an iterator over this mock dataset."""
        for i in range(self.length):
            yield self.sample


class repeatedly(IterableDataset, PipelineStage):
    """Repeatedly yield samples from a dataset."""

    def __init__(self, source, nepochs=None, nbatches=None, length=None):
        """Create an instance of Repeatedly.

        :param nepochs: repeat for a maximum of nepochs
        :param nbatches: repeat for a maximum of nbatches
        """
        self.source = source
        self.length = length
        self.nbatches = nbatches

    def invoke(self, source):
        """Return an iterator that iterates repeatedly over a source."""
        return utils.repeatedly(
            source,
            nepochs=self.nepochs,
            nbatches=self.nbatches, )


class with_epoch(IterableDataset):
    """Change the actual and nominal length of an IterableDataset.

    This will continuously iterate through the original dataset, but
    impose new epoch boundaries at the given length/nominal.
    This exists mainly as a workaround for the odd logic in DataLoader.
    It is also useful for choosing smaller nominal epoch sizes with
    very large datasets.

    """

    def __init__(self, dataset, length):
        """Chop the dataset to the given length.

        :param dataset: IterableDataset
        :param length: declared length of the dataset
        :param nominal: nominal length of dataset (if different from declared)
        """
        super().__init__()
        self.length = length
        self.source = None

    def __getstate__(self):
        """Return the pickled state of the dataset.

        This resets the dataset iterator, since that can't be pickled.
        """
        result = dict(self.__dict__)
        result["source"] = None
        return result

    def invoke(self, dataset):
        """Return an iterator over the dataset.

        This iterator returns as many samples as given by the `length`
        parameter.
        """
        if self.source is None:
            self.source = iter(dataset)
        for i in range(self.length):
            try:
                sample = next(self.source)
            except StopIteration:
                self.source = iter(dataset)
                try:
                    sample = next(self.source)
                except StopIteration:
                    return
            yield sample
        self.source = None


class with_length(IterableDataset, PipelineStage):
    """Repeatedly yield samples from a dataset."""

    def __init__(self, dataset, length):
        """Create an instance of Repeatedly.

        :param dataset: source dataset
        :param length: stated length
        """
        super().__init__()
        self.dataset = dataset
        self.length = length

    def invoke(self, dataset):
        """Return an iterator that iterates repeatedly over a source."""
        return iter(dataset)

    def __len__(self):
        """Return the user specified length."""
        return self.length
