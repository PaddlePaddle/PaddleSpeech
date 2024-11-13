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
    # _sos = paddle.to_tensor(
    #    [sos], dtype=ys_pad.dtype, stop_gradient=True, place=ys_pad.place)
    # _eos = paddle.to_tensor(
    #    [eos], dtype=ys_pad.dtype, stop_gradient=True, place=ys_pad.place)
    # ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    # ys_in = [paddle.concat([_sos, y], axis=0) for y in ys]
    # ys_out = [paddle.concat([y, _eos], axis=0) for y in ys]
    # return pad_sequence(ys_in, padding_value=eos).transpose([1,0]), pad_sequence(ys_out, padding_value=ignore_id).transpose([1,0])

    B = ys_pad.shape[0]
    _sos = paddle.full([B, 1], sos, dtype=ys_pad.dtype)
    _eos = paddle.full([B, 1], eos, dtype=ys_pad.dtype)
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


def reverse_pad_list(ys_pad: paddle.Tensor,
                     ys_lens: paddle.Tensor,
                     pad_value: float=-1.0) -> paddle.Tensor:
    """Reverse padding for the list of tensors.
    Args:
        ys_pad (tensor): The padded tensor (B, Tokenmax).
        ys_lens (tensor): The lens of token seqs (B)
        pad_value (int): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tokenmax).
    Examples:
        >>> x
        tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> pad_list(x, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])
    """
    r_ys_pad = pad_sequence([(paddle.flip(y.int()[:i], [0]))
                             for y, i in zip(ys_pad, ys_lens)], True, pad_value)
    return r_ys_pad


def st_reverse_pad_list(ys_pad: paddle.Tensor,
                        ys_lens: paddle.Tensor,
                        sos: float,
                        eos: float) -> paddle.Tensor:
    """Reverse padding for the list of tensors.
    Args:
        ys_pad (tensor): The padded tensor (B, Tokenmax).
        ys_lens (tensor): The lens of token seqs (B)
    Returns:
        Tensor: Padded tensor (B, Tokenmax).
    Examples:
        >>> x
        tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> pad_list(x, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])
    """
    # Equal to:
    #   >>> r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
    #   >>> r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
    B = ys_pad.shape[0]
    _sos = paddle.full([B, 1], sos, dtype=ys_pad.dtype)
    max_len = paddle.max(ys_lens)
    index_range = paddle.arange(0, max_len, 1)
    seq_len_expand = ys_lens.unsqueeze(1)
    seq_mask = seq_len_expand > index_range  # (beam, max_len)

    index = (seq_len_expand - 1) - index_range  # (beam, max_len)
    #   >>> index
    #   >>> tensor([[ 2,  1,  0],
    #   >>>         [ 2,  1,  0],
    #   >>>         [ 0, -1, -2]])
    index = index * seq_mask.astype(index.dtype)

    #   >>> index
    #   >>> tensor([[2, 1, 0],
    #   >>>         [2, 1, 0],
    #   >>>         [0, 0, 0]])
    def paddle_gather(x, dim, index):
        index_shape = index.shape
        index_flatten = index.flatten()
        if dim < 0:
            dim = len(x.shape) + dim
        nd_index = []
        for k in range(len(x.shape)):
            if k == dim:
                nd_index.append(index_flatten)
            else:
                reshape_shape = [1] * len(x.shape)
                reshape_shape[k] = x.shape[k]
                x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
                x_arange = x_arange.reshape(reshape_shape)
                dim_index = paddle.expand(x_arange, index_shape).flatten()
                nd_index.append(dim_index)
        ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
        paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
        return paddle_out

    r_hyps = paddle_gather(ys_pad, 1, index)
    #   >>> r_hyps
    #   >>> tensor([[3, 2, 1],
    #   >>>         [4, 8, 9],
    #   >>>         [2, 2, 2]])
    _eos = paddle.full([1], eos, dtype=r_hyps.dtype)
    r_hyps = paddle.where(seq_mask, r_hyps, _eos)
    #   >>> r_hyps
    #   >>> tensor([[3, 2, 1],
    #   >>>         [4, 8, 9],
    #   >>>         [2, eos, eos]])

    r_hyps = paddle.cat([_sos, r_hyps], dim=1)
    # r_hyps = paddle.concat([hyps[:, 0:1], r_hyps], axis=1)
    #   >>> r_hyps
    #   >>> tensor([[sos, 3, 2, 1],
    #   >>>         [sos, 4, 8, 9],
    #   >>>         [sos, 2, eos, eos]])
    return r_hyps
