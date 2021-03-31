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
"""Unility functions for Transformer."""
import math
import logging
from typing import Tuple, List

import paddle

logger = logging.getLogger(__name__)

__all__ = ["pad_list", "add_sos_eos", "th_accuracy"]

IGNORE_ID = -1


def pad_list(xs: List[paddle.Tensor], pad_value: int):
    """Perform padding for the list of tensors.
    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).
    Examples:
        >>> x = [paddle.ones(4), paddle.ones(2), paddle.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    max_len = max([x.size(0) for x in xs])
    pad = paddle.zeros(n_batch, max_len, dtype=xs[0].dtype)
    pad = pad.fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def add_sos_eos(ys_pad: paddle.Tensor, sos: int, eos: int,
                ignore_id: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Add <sos> and <eos> labels.
    Args:
        ys_pad (paddle.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding
    Returns:
        ys_in (paddle.Tensor) : (B, Lmax + 1)
        ys_out (paddle.Tensor) : (B, Lmax + 1)
    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=paddle.int32)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[10,  1,  2,  3,  4,  5],
                [10,  4,  5,  6, 11, 11],
                [10,  7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    _sos = paddle.to_tensor(
        [sos], dtype=paddle.long, stop_gradient=True, place=ys_pad.place)
    _eos = paddle.to_tensor(
        [eos], dtype=paddle.long, stop_gradient=True, place=ys_pad.place)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [paddle.cat([_sos, y], dim=0) for y in ys]
    ys_out = [paddle.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def th_accuracy(pad_outputs: paddle.Tensor,
                pad_targets: paddle.Tensor,
                ignore_label: int) -> float:
    """Calculate accuracy.
    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax, D).
        ignore_label (int): Ignore label id.
    Returns:
        float: Accuracy value (0.0 - 1.0).
    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = paddle.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = paddle.sum(mask)
    return float(numerator) / float(denominator)
