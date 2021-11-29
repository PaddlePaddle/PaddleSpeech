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
import paddle
from paddle import nn
from typeguard import check_argument_types


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Parameters
    ----------
    xs : List[Tensor]
        List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
    pad_value : float)
        Value for padding.

    Returns
    ----------
    Tensor
        Padded tensor (B, Tmax, `*`).

    Examples
    ----------
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

    Parameters
    ----------
    lengths : LongTensor
            Batch of lengths (B,).

    Returns
    ----------
    Tensor(bool)
        Mask tensor containing indices of padded part bool.

    Examples
    ----------
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

    Parameters
    ----------
    lengths : LongTensor or List
            Batch of lengths (B,).
    xs : Tensor, optional
        The reference tensor.
        If set, masks will be the same shape as this tensor.
    length_dim : int, optional
        Dimension indicator of the above tensor.
        See the example.

    Returns
    ----------
    Tensor(bool)
        mask tensor containing indices of padded part bool.

    Examples
    ----------
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

    Parameters
    ----------
    model : nn.Layer
        Target.
    init : str
        Method of initialization.
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
