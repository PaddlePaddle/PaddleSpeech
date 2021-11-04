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
# Modified from chainer(https://github.com/chainer/chainer)
import contextlib
import math
from collections import defaultdict

OBSERVATIONS = None


@contextlib.contextmanager
def ObsScope(observations):
    # make `observation` the target to report to.
    # it is basically a dictionary that stores temporary observations
    global OBSERVATIONS
    old = OBSERVATIONS
    OBSERVATIONS = observations

    try:
        yield
    finally:
        OBSERVATIONS = old


def get_observations():
    global OBSERVATIONS
    return OBSERVATIONS


def report(name, value):
    # a simple function to report named value
    # you can use it everywhere, it will get the default target and writ to it
    # you can think of it as std.out
    observations = get_observations()
    if observations is None:
        return
    else:
        observations[name] = value


class Summary():
    """Online summarization of a sequence of scalars.
    Summary computes the statistics of given scalars online.
    """

    def __init__(self):
        self._x = 0.0
        self._x2 = 0.0
        self._n = 0

    def add(self, value, weight=1):
        """Adds a scalar value.
        Args:
            value: Scalar value to accumulate. It is either a NumPy scalar or
                a zero-dimensional array (on CPU or GPU).
            weight: An optional weight for the value. It is a NumPy scalar or
                a zero-dimensional array (on CPU or GPU).
                Default is 1 (integer).
        """
        self._x += weight * value
        self._x2 += weight * value * value
        self._n += weight

    def compute_mean(self):
        """Computes the mean."""
        x, n = self._x, self._n
        return x / n

    def make_statistics(self):
        """Computes and returns the mean and standard deviation values.
        Returns:
            tuple: Mean and standard deviation values.
        """
        x, n = self._x, self._n
        mean = x / n
        var = self._x2 / n - mean * mean
        std = math.sqrt(var)
        return mean, std


class DictSummary():
    """Online summarization of a sequence of dictionaries.
    ``DictSummary`` computes the statistics of a given set of scalars online.
    It only computes the statistics for scalar values and variables of scalar
    values in the dictionaries.
    """

    def __init__(self):
        self._summaries = defaultdict(Summary)

    def add(self, d):
        """Adds a dictionary of scalars.
        Args:
            d (dict): Dictionary of scalars to accumulate. Only elements of
               scalars, zero-dimensional arrays, and variables of
               zero-dimensional arrays are accumulated. When the value
               is a tuple, the second element is interpreted as a weight.
        """
        summaries = self._summaries
        for k, v in d.items():
            w = 1
            if isinstance(v, tuple):
                v = v[0]
                w = v[1]
            summaries[k].add(v, weight=w)

    def compute_mean(self):
        """Creates a dictionary of mean values.
        It returns a single dictionary that holds a mean value for each entry
        added to the summary.
        Returns:
            dict: Dictionary of mean values.
        """
        return {
            name: summary.compute_mean()
            for name, summary in self._summaries.items()
        }

    def make_statistics(self):
        """Creates a dictionary of statistics.
        It returns a single dictionary that holds mean and standard deviation
        values for every entry added to the summary. For an entry of name
        ``'key'``, these values are added to the dictionary by names ``'key'``
        and ``'key.std'``, respectively.
        Returns:
            dict: Dictionary of statistics of all entries.
        """
        stats = {}
        for name, summary in self._summaries.items():
            mean, std = summary.make_statistics()
            stats[name] = mean
            stats[name + '.std'] = std

        return stats
