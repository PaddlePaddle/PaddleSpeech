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
from typing import List

import numpy as np
from numpy import ndarray as array

from ..backends import depth_convert
from ..utils import ParameterError

__all__ = [
    'depth_augment',
    'spect_augment',
    'random_crop1d',
    'random_crop2d',
    'adaptive_spect_augment',
]


def randint(high: int) -> int:
    """Generate one random integer in range [0 high)

     This is a helper function for random data augmentaiton
    """
    return int(np.random.randint(0, high=high))


def rand() -> float:
    """Generate one floating-point number in range [0 1)

    This is a helper function for random data augmentaiton
    """
    return float(np.random.rand(1))


def depth_augment(y: array,
                  choices: List=['int8', 'int16'],
                  probs: List[float]=[0.5, 0.5]) -> array:
    """ Audio depth augmentation

    Do audio depth augmentation to simulate the distortion brought by quantization.
    """
    assert len(probs) == len(
        choices
    ), 'number of choices {} must be equal to size of probs {}'.format(
        len(choices), len(probs))
    depth = np.random.choice(choices, p=probs)
    src_depth = y.dtype
    y1 = depth_convert(y, depth)
    y2 = depth_convert(y1, src_depth)

    return y2


def adaptive_spect_augment(spect: array, tempo_axis: int=0,
                           level: float=0.1) -> array:
    """Do adpative spectrogram augmentation

    The level of the augmentation is gowern by the paramter level,
    ranging from 0 to 1, with 0 represents no augmentation。

    """
    assert spect.ndim == 2., 'only supports 2d tensor or numpy array'
    if tempo_axis == 0:
        nt, nf = spect.shape
    else:
        nf, nt = spect.shape

    time_mask_width = int(nt * level * 0.5)
    freq_mask_width = int(nf * level * 0.5)

    num_time_mask = int(10 * level)
    num_freq_mask = int(10 * level)

    if tempo_axis == 0:
        for _ in range(num_time_mask):
            start = randint(nt - time_mask_width)
            spect[start:start + time_mask_width, :] = 0
        for _ in range(num_freq_mask):
            start = randint(nf - freq_mask_width)
            spect[:, start:start + freq_mask_width] = 0
    else:
        for _ in range(num_time_mask):
            start = randint(nt - time_mask_width)
            spect[:, start:start + time_mask_width] = 0
        for _ in range(num_freq_mask):
            start = randint(nf - freq_mask_width)
            spect[start:start + freq_mask_width, :] = 0

    return spect


def spect_augment(spect: array,
                  tempo_axis: int=0,
                  max_time_mask: int=3,
                  max_freq_mask: int=3,
                  max_time_mask_width: int=30,
                  max_freq_mask_width: int=20) -> array:
    """Do spectrogram augmentation in both time and freq axis

    Reference:

    """
    assert spect.ndim == 2., 'only supports 2d tensor or numpy array'
    if tempo_axis == 0:
        nt, nf = spect.shape
    else:
        nf, nt = spect.shape

    num_time_mask = randint(max_time_mask)
    num_freq_mask = randint(max_freq_mask)

    time_mask_width = randint(max_time_mask_width)
    freq_mask_width = randint(max_freq_mask_width)

    if tempo_axis == 0:
        for _ in range(num_time_mask):
            start = randint(nt - time_mask_width)
            spect[start:start + time_mask_width, :] = 0
        for _ in range(num_freq_mask):
            start = randint(nf - freq_mask_width)
            spect[:, start:start + freq_mask_width] = 0
    else:
        for _ in range(num_time_mask):
            start = randint(nt - time_mask_width)
            spect[:, start:start + time_mask_width] = 0
        for _ in range(num_freq_mask):
            start = randint(nf - freq_mask_width)
            spect[start:start + freq_mask_width, :] = 0

    return spect


def random_crop1d(y: array, crop_len: int) -> array:
    """ Do random cropping on 1d input signal

    The input is a 1d signal, typically a sound waveform
    """
    if y.ndim != 1:
        'only accept 1d tensor or numpy array'
    n = len(y)
    idx = randint(n - crop_len)
    return y[idx:idx + crop_len]


def random_crop2d(s: array, crop_len: int, tempo_axis: int=0) -> array:
    """ Do random cropping for 2D array, typically a spectrogram.

    The cropping is done in temporal direction on the time-freq input signal.
    """
    if tempo_axis >= s.ndim:
        raise ParameterError('axis out of range')

    n = s.shape[tempo_axis]
    idx = randint(high=n - crop_len)
    sli = [slice(None) for i in range(s.ndim)]
    sli[tempo_axis] = slice(idx, idx + crop_len)
    out = s[tuple(sli)]
    return out
