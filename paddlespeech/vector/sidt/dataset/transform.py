#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2020    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################

"""
Transform operator for audio features
"""

import sys
import random
import numpy as np

import sidt.utils.utils as utils
from sidt import _logger as log


class BaseTrans(object):
    """
    Base class of all transforms used in audio feature
    """
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self._apply(inputs)

    def _apply(self, inputs):
        raise NotImplementedError


class RandomFlip(BaseTrans):
    """
    Flip the input data randomly with a given probability.
    """
    def __init__(self, prob, axis=1):
        self._prob = prob
        self._axis = axis

    def _apply(self, inputs):
        if random.random() < self._prob:
            return np.flip(inputs, axis=self._axis)
        else:
            return inputs


class RandomBlockShuffle(BaseTrans):
    """
    Shuffle a block data of the input data randomly with a given probability.
    """
    def __init__(self, prob=0.5, scale=(0.1, 0.5), axis=1):
        self._prob = prob
        self._scale = scale
        self._axis = axis
        assert axis >= 0 and axis < 2

    def _apply(self, inputs):
        if random.random() < self._prob:
            data_size = inputs.shape[self._axis]
            block_size = random.randint(int(data_size * self._scale[0]), int(data_size * self._scale[1]))
            bg = random.randint(0, data_size - block_size)
            data = np.zeros_like(inputs)
            if self._axis:
                data[:, 0:bg] = inputs[:, block_size:block_size + bg]
                data[:, bg:bg + block_size] = inputs[:, 0:block_size]
                data[:, bg + block_size:] = inputs[:, bg + block_size:]
            else:
                data[0:bg, :] = inputs[block_size:block_size + bg, :]
                data[bg:bg + block_size, :] = inputs[0:block_size, :]
                data[bg + block_size:, :] = inputs[bg + block_size:, :]
            return data
        else:
            return inputs


class RandomErasing(BaseTrans):
    """
    Erase a little block of the input data randomly with a given probability.
    """
    def __init__(self, prob=0.5, scale=(0.01, 0.05), value=0.0, axis=1):
        self._prob = prob
        self._scale = scale
        self._value = value
        self._axis = axis
        assert axis >= 0 and axis < 2

    def _apply(self, inputs):
        if random.random() < self._prob:
            data_size = inputs.shape[self._axis]
            bg = random.randint(0, data_size - 1)
            end = min(data_size,
                      bg + random.randint(int(data_size * self._scale[0]), int(data_size * self._scale[1])))
            if self._axis:
                inputs[:, bg:end] = self._value
            else:
                inputs[bg:end, :] = self._value
            return inputs
        else:
            return inputs



if __name__ == "__main__":
    from sidt.utils.seed import seed_everything
    seed_everything(0)

    inputs = np.random.rand(4, 20)
    a = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

    flip_a = np.array([[6, 5, 4, 3, 2, 1], [12, 11, 10, 9, 8, 7]])
    t = RandomFlip(1)
    print(t(a))
    assert np.all(np.equal(flip_a, t(a)))

    shuf_a = np.array([[4, 5, 6, 1, 2, 3], [10, 11, 12, 7, 8, 9]])
    t = RandomBlockShuffle(1, (0.5, 0.5))
    print(t(a))
    assert np.all(np.equal(shuf_a, t(a)))

    t = RandomErasing(1, (0.1, 0.4))
    print(t(a))
