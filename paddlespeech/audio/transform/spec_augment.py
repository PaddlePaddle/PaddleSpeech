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
# Modified from espnet(https://github.com/espnet/espnet)
"""Spec Augment module for preprocessing i.e., data augmentation"""
import random

import numpy
from PIL import Image

from .functional import FuncTrans


def time_warp(x, max_time_warp=80, inplace=False, mode="PIL"):
    """time warp for spec augment

    move random center frame by the random width ~ uniform(-window, window)
    :param numpy.ndarray x: spectrogram (time, freq)
    :param int max_time_warp: maximum time frames to warp
    :param bool inplace: overwrite x with the result
    :param str mode: "PIL" (default, fast, not differentiable) or "sparse_image_warp"
        (slow, differentiable)
    :returns numpy.ndarray: time warped spectrogram (time, freq)
    """
    window = max_time_warp
    if window == 0:
        return x

    if mode == "PIL":
        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
        center = random.randrange(window, t - window)
        warped = random.randrange(center - window,
                                  center + window) + 1  # 1 ... t - 1

        left = Image.fromarray(x[:center]).resize((x.shape[1], warped),
                                                  Image.BICUBIC)
        right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped),
                                                   Image.BICUBIC)
        if inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return numpy.concatenate((left, right), 0)
    elif mode == "sparse_image_warp":
        import paddle

        from espnet.utils import spec_augment

        # TODO(karita): make this differentiable again
        return spec_augment.time_warp(paddle.to_tensor(x), window).numpy()
    else:
        raise NotImplementedError("unknown resize mode: " + mode +
                                  ", choose one from (PIL, sparse_image_warp).")


class TimeWarp(FuncTrans):
    _func = time_warp
    __doc__ = time_warp.__doc__

    def __call__(self, x, train):
        if not train:
            return x
        return super().__call__(x)


def freq_mask(x, F=30, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument

    :param numpy.ndarray x: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = x
    else:
        cloned = x.copy()

    num_mel_channels = cloned.shape[1]
    fs = numpy.random.randint(0, F, size=(n_mask, 2))

    for f, mask_end in fs:
        f_zero = random.randrange(0, num_mel_channels - f)
        mask_end += f_zero

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            continue

        if replace_with_zero:
            cloned[:, f_zero:mask_end] = 0
        else:
            cloned[:, f_zero:mask_end] = cloned.mean()
    return cloned


class FreqMask(FuncTrans):
    _func = freq_mask
    __doc__ = freq_mask.__doc__

    def __call__(self, x, train):
        if not train:
            return x
        return super().__call__(x)


def time_mask(spec, T=40, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument

    :param numpy.ndarray spec: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = spec
    else:
        cloned = spec.copy()
    len_spectro = cloned.shape[0]
    ts = numpy.random.randint(0, T, size=(n_mask, 2))
    for t, mask_end in ts:
        # avoid randint range error
        if len_spectro - t <= 0:
            continue
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            continue

        mask_end += t_zero
        if replace_with_zero:
            cloned[t_zero:mask_end] = 0
        else:
            cloned[t_zero:mask_end] = cloned.mean()
    return cloned


class TimeMask(FuncTrans):
    _func = time_mask
    __doc__ = time_mask.__doc__

    def __call__(self, x, train):
        if not train:
            return x
        return super().__call__(x)


def spec_augment(
    x,
    resize_mode="PIL",
    max_time_warp=80,
    max_freq_width=27,
    n_freq_mask=2,
    max_time_width=100,
    n_time_mask=2,
    inplace=True,
    replace_with_zero=True,
):
    """spec agument

    apply random time warping and time/freq masking
    default setting is based on LD (Librispeech double) in Table 2
        https://arxiv.org/pdf/1904.08779.pdf

    :param numpy.ndarray x: (time, freq)
    :param str resize_mode: "PIL" (fast, nondifferentiable) or "sparse_image_warp"
        (slow, differentiable)
    :param int max_time_warp: maximum frames to warp the center frame in spectrogram (W)
    :param int freq_mask_width: maximum width of the random freq mask (F)
    :param int n_freq_mask: the number of the random freq mask (m_F)
    :param int time_mask_width: maximum width of the random time mask (T)
    :param int n_time_mask: the number of the random time mask (m_T)
    :param bool inplace: overwrite intermediate array
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    assert isinstance(x, numpy.ndarray)
    assert x.ndim == 2
    x = time_warp(x, max_time_warp, inplace=inplace, mode=resize_mode)
    x = freq_mask(
        x,
        max_freq_width,
        n_freq_mask,
        inplace=inplace,
        replace_with_zero=replace_with_zero,
    )
    x = time_mask(
        x,
        max_time_width,
        n_time_mask,
        inplace=inplace,
        replace_with_zero=replace_with_zero,
    )
    return x


class SpecAugment(FuncTrans):
    _func = spec_augment
    __doc__ = spec_augment.__doc__

    def __call__(self, x, train):
        if not train:
            return x
        return super().__call__(x)
