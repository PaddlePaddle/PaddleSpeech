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
from typing import List
from typing import Tuple

import paddle

from .log import Logger

__all__ = ["pad_sequence", "add_sos_eos", "th_accuracy", "has_tensor"]

logger = Logger(__name__)


def has_tensor(val):
    if isinstance(val, (list, tuple)):
        for item in val:
            if has_tensor(item):
                return True
    elif isinstance(val, dict):
        for k, v in val.items():
            print(k)
            if has_tensor(v):
                return True
    else:
        return paddle.is_tensor(val)


def pad_sequence(sequences: List[paddle.Tensor],
                 batch_first: bool=False,
                 padding_value: float=0.0) -> paddle.Tensor:
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from paddle.nn.utils.rnn import pad_sequence
        >>> a = paddle.ones(25, 300)
        >>> b = paddle.ones(22, 300)
        >>> c = paddle.ones(15, 300)
        >>> pad_sequence([a, b, c]).shape
        paddle.Tensor([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = paddle.shape(sequences[0])
    # (TODO Hui Zhang): slice not supprot `end==start`
    # trailing_dims = max_size[1:]
    trailing_dims = tuple(
        max_size[1:].numpy().tolist()) if sequences[0].ndim >= 2 else ()
    max_len = max([s.shape[0] for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    out_tensor = paddle.full(out_dims, padding_value, sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            # TODO (Hui Zhang): set_value op not supprot `end==start`
            # TODO (Hui Zhang): set_value op not support int16
            # TODO (Hui Zhang): set_varbase 2 rank not support [0,0,...]
            # out_tensor[i, :length, ...] = tensor
            if length != 0:
                out_tensor[i, :length] = tensor
            else:
                out_tensor[i, length] = tensor
        else:
            # TODO (Hui Zhang): set_value op not supprot `end==start`
            # out_tensor[:length, i, ...] = tensor
            if length != 0:
                out_tensor[:length, i] = tensor
            else:
                out_tensor[length, i] = tensor

    return out_tensor


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
    # TODO(Hui Zhang): using comment code,
    #_sos = paddle.to_tensor(
    #    [sos], dtype=paddle.long, stop_gradient=True, place=ys_pad.place)
    #_eos = paddle.to_tensor(
    #    [eos], dtype=paddle.long, stop_gradient=True, place=ys_pad.place)
    #ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    #ys_in = [paddle.cat([_sos, y], dim=0) for y in ys]
    #ys_out = [paddle.cat([y, _eos], dim=0) for y in ys]
    #return pad_sequence(ys_in, padding_value=eos), pad_sequence(ys_out, padding_value=ignore_id)
    B = ys_pad.shape[0]
    _sos = paddle.ones([B, 1], dtype=ys_pad.dtype) * sos
    _eos = paddle.ones([B, 1], dtype=ys_pad.dtype) * eos
    ys_in = paddle.cat([_sos, ys_pad], dim=1)
    mask_pad = (ys_in == ignore_id)
    ys_in = ys_in.masked_fill(mask_pad, eos)

    ys_out = paddle.cat([ys_pad, _eos], dim=1)
    ys_out = ys_out.masked_fill(mask_pad, eos)
    mask_eos = (ys_out == ignore_id)
    ys_out = ys_out.masked_fill(mask_eos, eos)
    ys_out = ys_out.masked_fill(mask_pad, ignore_id)
    return ys_in, ys_out


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
    pad_pred = pad_outputs.reshape(
        [pad_targets.shape[0], pad_targets.shape[1],
         pad_outputs.shape[1]]).argmax(2)
    mask = pad_targets != ignore_label
    #TODO(Hui Zhang): sum not support bool type
    # numerator = paddle.sum(
    #     pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    numerator = (
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    numerator = paddle.sum(numerator.type_as(pad_targets))
    #TODO(Hui Zhang): sum not support bool type
    # denominator = paddle.sum(mask)
    denominator = paddle.sum(mask.type_as(pad_targets))
    return float(numerator) / float(denominator)
