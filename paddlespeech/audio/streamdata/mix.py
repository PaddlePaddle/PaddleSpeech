#
# Copyright (c) 2017-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
# Modified from https://github.com/webdataset/webdataset
#
"""Classes for mixing samples from multiple sources."""
import itertools
import os
import random
import sys
import time
from functools import reduce
from functools import wraps

import numpy as np

from . import autodecode
from . import utils
from .paddle_utils import IterableDataset
from .paddle_utils import PaddleTensor
from .utils import PipelineStage


def round_robin_shortest(*sources):
    i = 0
    while True:
        try:
            sample = next(sources[i % len(sources)])
            yield sample
        except StopIteration:
            break
        i += 1


def round_robin_longest(*sources):
    i = 0
    while len(sources) > 0:
        try:
            sample = next(sources[i])
            i += 1
            yield sample
        except StopIteration:
            del sources[i]


class RoundRobin(IterableDataset):
    def __init__(self, datasets, longest=False):
        self.datasets = datasets
        self.longest = longest

    def __iter__(self):
        """Return an iterator over the sources."""
        sources = [iter(d) for d in self.datasets]
        if self.longest:
            return round_robin_longest(*sources)
        else:
            return round_robin_shortest(*sources)


def random_samples(sources, probs=None, longest=False):
    if probs is None:
        probs = [1] * len(sources)
    else:
        probs = list(probs)
    while len(sources) > 0:
        cum = (np.array(probs) / np.sum(probs)).cumsum()
        r = random.random()
        i = np.searchsorted(cum, r)
        try:
            yield next(sources[i])
        except StopIteration:
            if longest:
                del sources[i]
                del probs[i]
            else:
                break


class RandomMix(IterableDataset):
    def __init__(self, datasets, probs=None, longest=False):
        self.datasets = datasets
        self.probs = probs
        self.longest = longest

    def __iter__(self):
        """Return an iterator over the sources."""
        sources = [iter(d) for d in self.datasets]
        return random_samples(sources, self.probs, longest=self.longest)
