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
import logging
from typing import Union
from typing import Optional
from typing import List
from typing import Tuple
from typing import Any

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

logger = logging.getLogger(__name__)

########### hcak logging #############
logger.warn = logging.warning

########### hcak paddle #############
paddle.bool = 'bool'
paddle.float16 = 'float16'
paddle.half = 'float16'
paddle.float32 = 'float32'
paddle.float = 'float32'
paddle.float64 = 'float64'
paddle.double = 'float64'
paddle.int8 = 'int8'
paddle.int16 = 'int16'
paddle.short = 'int16'
paddle.int32 = 'int32'
paddle.int = 'int32'
paddle.int64 = 'int64'
paddle.long = 'int64'
paddle.uint8 = 'uint8'
paddle.complex64 = 'complex64'
paddle.complex128 = 'complex128'
paddle.cdouble = 'complex128'

if not hasattr(paddle, 'softmax'):
    logger.warn("register user softmax to paddle, remove this when fixed!")
    setattr(paddle, 'softmax', paddle.nn.functional.softmax)

if not hasattr(paddle, 'log_softmax'):
    logger.warn("register user log_softmax to paddle, remove this when fixed!")
    setattr(paddle, 'log_softmax', paddle.nn.functional.log_softmax)

if not hasattr(paddle, 'sigmoid'):
    logger.warn("register user sigmoid to paddle, remove this when fixed!")
    setattr(paddle, 'sigmoid', paddle.nn.functional.sigmoid)

if not hasattr(paddle, 'log_sigmoid'):
    logger.warn("register user log_sigmoid to paddle, remove this when fixed!")
    setattr(paddle, 'log_sigmoid', paddle.nn.functional.log_sigmoid)

if not hasattr(paddle, 'relu'):
    logger.warn("register user relu to paddle, remove this when fixed!")
    setattr(paddle, 'relu', paddle.nn.functional.relu)


def cat(xs, dim=0):
    return paddle.concat(xs, axis=dim)


if not hasattr(paddle, 'cat'):
    logger.warn(
        "override cat of paddle if exists or register, remove this when fixed!")
    paddle.cat = cat


########### hcak paddle.Tensor #############
def item(x: paddle.Tensor):
    if x.dtype == paddle.fluid.core_avx.VarDesc.VarType.FP32:
        return float(x)
    else:
        raise ValueError("not support")


if not hasattr(paddle.Tensor, 'item'):
    logger.warn(
        "override item of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.item = item


def func_long(x: paddle.Tensor):
    return paddle.cast(x, paddle.long)


if not hasattr(paddle.Tensor, 'long'):
    logger.warn(
        "override long of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.long = func_long

if not hasattr(paddle.Tensor, 'numel'):
    logger.warn(
        "override numel of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.numel = paddle.numel


def new_full(x: paddle.Tensor,
             size: Union[List[int], Tuple[int], paddle.Tensor],
             fill_value: Union[float, int, bool, paddle.Tensor],
             dtype=None):
    return paddle.full(size, fill_value, dtype=x.dtype)


if not hasattr(paddle.Tensor, 'new_full'):
    logger.warn(
        "override new_full of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.new_full = new_full


def eq(xs: paddle.Tensor, ys: Union[paddle.Tensor, float]) -> paddle.Tensor:
    return xs.equal(paddle.to_tensor(ys, dtype=xs.dtype, place=xs.place))


if not hasattr(paddle.Tensor, 'eq'):
    logger.warn(
        "override eq of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.eq = eq


def contiguous(xs: paddle.Tensor) -> paddle.Tensor:
    return xs


if not hasattr(paddle.Tensor, 'contiguous'):
    logger.warn(
        "override contiguous of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.contiguous = contiguous


def size(xs: paddle.Tensor, *args: int) -> paddle.Tensor:
    nargs = len(args)
    assert (nargs <= 1)
    s = paddle.shape(xs)
    if nargs == 1:
        return s[args[0]]
    else:
        return s


#`to_static` do not process `size` property, maybe some `paddle` api dependent on it.
logger.warn(
    "override size of paddle.Tensor "
    "(`to_static` do not process `size` property, maybe some `paddle` api dependent on it), remove this when fixed!"
)
paddle.Tensor.size = size


def view(xs: paddle.Tensor, *args: int) -> paddle.Tensor:
    return xs.reshape(args)


if not hasattr(paddle.Tensor, 'view'):
    logger.warn("register user view to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.view = view


def view_as(xs: paddle.Tensor, ys: paddle.Tensor) -> paddle.Tensor:
    return xs.reshape(ys.size())


if not hasattr(paddle.Tensor, 'view_as'):
    logger.warn(
        "register user view_as to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.view_as = view_as


def masked_fill(xs: paddle.Tensor,
                mask: paddle.Tensor,
                value: Union[float, int]):
    assert xs.shape == mask.shape
    trues = paddle.ones_like(xs) * value
    xs = paddle.where(mask, trues, xs)
    return xs


if not hasattr(paddle.Tensor, 'masked_fill'):
    logger.warn(
        "register user masked_fill to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.masked_fill = masked_fill


def masked_fill_(xs: paddle.Tensor,
                 mask: paddle.Tensor,
                 value: Union[float, int]):
    assert xs.shape == mask.shape
    trues = paddle.ones_like(xs) * value
    ret = paddle.where(mask, trues, xs)
    paddle.assign(ret, output=xs)


if not hasattr(paddle.Tensor, 'masked_fill_'):
    logger.warn(
        "register user masked_fill_ to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.masked_fill_ = masked_fill_


def fill_(xs: paddle.Tensor, value: Union[float, int]):
    val = paddle.full_like(xs, value)
    paddle.assign(val, output=xs)


if not hasattr(paddle.Tensor, 'fill_'):
    logger.warn("register user fill_ to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.fill_ = fill_


def repeat(xs: paddle.Tensor, *size: Any) -> paddle.Tensor:
    return paddle.tile(xs, size)


if not hasattr(paddle.Tensor, 'repeat'):
    logger.warn(
        "register user repeat to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.repeat = repeat

if not hasattr(paddle.Tensor, 'softmax'):
    logger.warn(
        "register user softmax to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'softmax', paddle.nn.functional.softmax)

if not hasattr(paddle.Tensor, 'sigmoid'):
    logger.warn(
        "register user sigmoid to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'sigmoid', paddle.nn.functional.sigmoid)

if not hasattr(paddle.Tensor, 'relu'):
    logger.warn("register user relu to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'relu', paddle.nn.functional.relu)


def type_as(x: paddle.Tensor, other: paddle.Tensor) -> paddle.Tensor:
    return x.astype(other.dtype)


if not hasattr(paddle.Tensor, 'type_as'):
    logger.warn(
        "register user type_as to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'type_as', type_as)


def to(x: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
    assert len(args) == 1
    if isinstace(args[0], str):  # dtype
        return x.astype(args[0])
    elif isinstance(args[0], paddle.Tensor):  #Tensor
        return x.astype(args[0].dtype)
    else:  # Device
        return x


if not hasattr(paddle.Tensor, 'to'):
    logger.warn("register user to to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'to', to)


def func_float(x: paddle.Tensor) -> paddle.Tensor:
    return x.astype(paddle.float)


if not hasattr(paddle.Tensor, 'float'):
    logger.warn("register user float to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'float', func_float)

########### hcak paddle.nn.functional #############


def glu(x: paddle.Tensor, dim=-1) -> paddle.Tensor:
    """The gated linear unit (GLU) activation."""
    a, b = x.split(2, axis=dim)
    act_b = F.sigmoid(b)
    return a * act_b


if not hasattr(paddle.nn.functional, 'glu'):
    logger.warn(
        "register user glu to paddle.nn.functional, remove this when fixed!")
    setattr(paddle.nn.functional, 'glu', glu)

# def softplus(x):
#     """Softplus function."""
#     if hasattr(paddle.nn.functional, 'softplus'):
#         #return paddle.nn.functional.softplus(x.float()).type_as(x)
#         return paddle.nn.functional.softplus(x)
#     else:
#         raise NotImplementedError

# def gelu_accurate(x):
#     """Gaussian Error Linear Units (GELU) activation."""
#     # [reference] https://github.com/pytorch/fairseq/blob/e75cff5f2c1d62f12dc911e0bf420025eb1a4e33/fairseq/modules/gelu.py
#     if not hasattr(gelu_accurate, "_a"):
#         gelu_accurate._a = math.sqrt(2 / math.pi)
#     return 0.5 * x * (1 + paddle.tanh(gelu_accurate._a *
#                                       (x + 0.044715 * paddle.pow(x, 3))))

# def gelu(x):
#     """Gaussian Error Linear Units (GELU) activation."""
#     if hasattr(nn.functional, 'gelu'):
#         #return nn.functional.gelu(x.float()).type_as(x)
#         return nn.functional.gelu(x)
#     else:
#         return x * 0.5 * (1.0 + paddle.erf(x / math.sqrt(2.0)))


# hack loss
def ctc_loss(logits,
             labels,
             input_lengths,
             label_lengths,
             blank=0,
             reduction='mean',
             norm_by_times=True):
    #logger.info("my ctc loss with norm by times")
    ## https://github.com/PaddlePaddle/Paddle/blob/f5ca2db2cc/paddle/fluid/operators/warpctc_op.h#L403
    loss_out = paddle.fluid.layers.warpctc(logits, labels, blank, norm_by_times,
                                           input_lengths, label_lengths)

    loss_out = paddle.fluid.layers.squeeze(loss_out, [-1])
    logger.info(f"warpctc loss: {loss_out}/{loss_out.shape} ")
    assert reduction in ['mean', 'sum', 'none']
    if reduction == 'mean':
        loss_out = paddle.mean(loss_out / label_lengths)
    elif reduction == 'sum':
        loss_out = paddle.sum(loss_out)
    logger.info(f"ctc loss: {loss_out}")
    return loss_out


logger.warn(
    "override ctc_loss of paddle.nn.functional if exists, remove this when fixed!"
)
F.ctc_loss = ctc_loss

########### hcak paddle.nn #############
if not hasattr(paddle.nn, 'Module'):
    logger.warn("register user Module to paddle.nn, remove this when fixed!")
    setattr(paddle.nn, 'Module', paddle.nn.Layer)

# maybe cause assert isinstance(sublayer, core.Layer)
if not hasattr(paddle.nn, 'ModuleList'):
    logger.warn(
        "register user ModuleList to paddle.nn, remove this when fixed!")
    setattr(paddle.nn, 'ModuleList', paddle.nn.LayerList)


class GLU(nn.Layer):
    """Gated Linear Units (GLU) Layer"""

    def __init__(self, dim: int=-1):
        super().__init__()
        self.dim = dim

    def forward(self, xs):
        return glu(xs, dim=self.dim)


if not hasattr(paddle.nn, 'GLU'):
    logger.warn("register user GLU to paddle.nn, remove this when fixed!")
    setattr(paddle.nn, 'GLU', GLU)


# TODO(Hui Zhang): remove this Layer
class ConstantPad2d(nn.Layer):
    """Pads the input tensor boundaries with a constant value.
    For N-dimensional padding, use paddle.nn.functional.pad().
    """

    def __init__(self, padding: Union[tuple, list, int], value: float):
        """
        Args:
            paddle ([tuple]): the size of the padding. 
                If is int, uses the same padding in all boundaries. 
                If a 4-tuple, uses (padding_left, padding_right, padding_top, padding_bottom)
            value ([flaot]): pad value
        """
        self.padding = padding if isinstance(padding,
                                             [tuple, list]) else [padding] * 4
        self.value = value

    def forward(self, xs: paddle.Tensor) -> paddle.Tensor:
        return nn.functional.pad(
            xs,
            self.padding,
            mode='constant',
            value=self.value,
            data_format='NCHW')


if not hasattr(paddle.nn, 'ConstantPad2d'):
    logger.warn(
        "register user ConstantPad2d to paddle.nn, remove this when fixed!")
    setattr(paddle.nn, 'ConstantPad2d', ConstantPad2d)

########### hcak paddle.jit #############

if not hasattr(paddle.jit, 'export'):
    logger.warn("register user export to paddle.jit, remove this when fixed!")
    setattr(paddle.jit, 'export', paddle.jit.to_static)


########### hcak paddle.nn.utils #############
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
        >>> pad_sequence([a, b, c]).size()
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
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


if not hasattr(paddle.nn.utils, 'rnn.pad_sequence'):
    logger.warn(
        "register user rnn.pad_sequence to paddle.nn.utils, remove this when fixed!"
    )
    setattr(paddle.nn.utils, 'rnn.pad_sequence', pad_sequence)
