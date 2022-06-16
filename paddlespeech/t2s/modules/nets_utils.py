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
from typing import Tuple

import paddle
from paddle import nn
from typeguard import check_argument_types


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List[Tensor]): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [paddle.ones([4]), paddle.ones([2]), paddle.ones([1])]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    max_len = max(x.shape[0] for x in xs)
    pad = paddle.full([n_batch, max_len, *xs[0].shape[1:]], pad_value)

    for i in range(n_batch):
        pad[i, :xs[i].shape[0]] = xs[i]

    return pad


def make_pad_mask(lengths, length_dim=-1):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (Tensor(int64)): Batch of lengths (B,).

    Returns: 
        Tensor(bool): Mask tensor containing indices of padded part bool.

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 1, 1]]
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    bs = paddle.shape(lengths)[0]
    maxlen = lengths.max()
    seq_range = paddle.arange(0, maxlen, dtype=paddle.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand([bs, maxlen])
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask


def make_non_pad_mask(lengths, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (Tensor(int64) or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor(bool): mask tensor containing indices of padded part bool.

    Examples: 
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                    [1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0]]
    """
    return paddle.logical_not(make_pad_mask(lengths, length_dim))


def initialize(model: nn.Layer, init: str):
    """Initialize weights of a neural network module.

    Parameters are initialized using the given method or distribution.

    Custom initialization routines can be implemented into submodules

    Args:
        model (nn.Layer): Target.
        init (str): Method of initialization.
    """
    assert check_argument_types()

    if init == "xavier_uniform":
        nn.initializer.set_global_initializer(nn.initializer.XavierUniform(),
                                              nn.initializer.Constant())
    elif init == "xavier_normal":
        nn.initializer.set_global_initializer(nn.initializer.XavierNormal(),
                                              nn.initializer.Constant())
    elif init == "kaiming_uniform":
        nn.initializer.set_global_initializer(nn.initializer.KaimingUniform(),
                                              nn.initializer.Constant())
    elif init == "kaiming_normal":
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(),
                                              nn.initializer.Constant())
    else:
        raise ValueError("Unknown initialization: " + init)


# for VITS
def get_random_segments(
        x: paddle.paddle,
        x_lengths: paddle.Tensor,
        segment_size: int, ) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Get random segments.
    Args:
        x (Tensor): Input tensor (B, C, T).
        x_lengths (Tensor): Length tensor (B,).
        segment_size (int): Segment size.
    Returns:
        Tensor: Segmented tensor (B, C, segment_size).
        Tensor: Start index tensor (B,).
    """
    b, c, t = paddle.shape(x)
    max_start_idx = x_lengths - segment_size
    start_idxs = paddle.cast(paddle.rand([b]) * max_start_idx, 'int64')
    segments = get_segments(x, start_idxs, segment_size)

    return segments, start_idxs


def get_segments(
        x: paddle.Tensor,
        start_idxs: paddle.Tensor,
        segment_size: int, ) -> paddle.Tensor:
    """Get segments.
    Args:
        x (Tensor): Input tensor (B, C, T).
        start_idxs (Tensor): Start index tensor (B,).
        segment_size (int): Segment size.
    Returns:
        Tensor: Segmented tensor (B, C, segment_size).
    """
    b, c, t = paddle.shape(x)
    segments = paddle.zeros([b, c, segment_size], dtype=x.dtype)
    for i, start_idx in enumerate(start_idxs):
        segments[i] = x[i, :, start_idx:start_idx + segment_size]
    return segments


# see https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/pytorch_project_convertor/API_docs/ops/torch.gather.md
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
