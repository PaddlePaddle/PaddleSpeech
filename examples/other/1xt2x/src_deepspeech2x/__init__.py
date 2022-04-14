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
from typing import Any
from typing import List
from typing import Tuple
from typing import Union

import paddle
from paddle import nn
from paddle.fluid import core
from paddle.nn import functional as F

from paddlespeech.s2t.utils.log import Log

#TODO(Hui Zhang): remove  fluid import
logger = Log(__name__).getlog()

########### hack logging #############
logger.warn = logger.warning

########### hack paddle #############
paddle.half = 'float16'
paddle.float = 'float32'
paddle.double = 'float64'
paddle.short = 'int16'
paddle.int = 'int32'
paddle.long = 'int64'
paddle.uint16 = 'uint16'
paddle.cdouble = 'complex128'


def convert_dtype_to_string(tensor_dtype):
    """
    Convert the data type in numpy to the data type in Paddle
    Args:
        tensor_dtype(core.VarDesc.VarType): the data type in numpy.
    Returns:
        core.VarDesc.VarType: the data type in Paddle.
    """
    dtype = tensor_dtype
    if dtype == core.VarDesc.VarType.FP32:
        return paddle.float32
    elif dtype == core.VarDesc.VarType.FP64:
        return paddle.float64
    elif dtype == core.VarDesc.VarType.FP16:
        return paddle.float16
    elif dtype == core.VarDesc.VarType.INT32:
        return paddle.int32
    elif dtype == core.VarDesc.VarType.INT16:
        return paddle.int16
    elif dtype == core.VarDesc.VarType.INT64:
        return paddle.int64
    elif dtype == core.VarDesc.VarType.BOOL:
        return paddle.bool
    elif dtype == core.VarDesc.VarType.BF16:
        # since there is still no support for bfloat16 in NumPy,
        # uint16 is used for casting bfloat16
        return paddle.uint16
    elif dtype == core.VarDesc.VarType.UINT8:
        return paddle.uint8
    elif dtype == core.VarDesc.VarType.INT8:
        return paddle.int8
    elif dtype == core.VarDesc.VarType.COMPLEX64:
        return paddle.complex64
    elif dtype == core.VarDesc.VarType.COMPLEX128:
        return paddle.complex128
    else:
        raise ValueError("Not supported tensor dtype %s" % dtype)


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


########### hack paddle.Tensor #############
def item(x: paddle.Tensor):
    return x.numpy().item()


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
    if convert_dtype_to_string(xs.dtype) == paddle.bool:
        xs = xs.astype(paddle.int)
    return xs.equal(
        paddle.to_tensor(
            ys, dtype=convert_dtype_to_string(xs.dtype), place=xs.place))


if not hasattr(paddle.Tensor, 'eq'):
    logger.warn(
        "override eq of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.eq = eq

if not hasattr(paddle, 'eq'):
    logger.warn(
        "override eq of paddle if exists or register, remove this when fixed!")
    paddle.eq = eq


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


def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def masked_fill(xs: paddle.Tensor,
                mask: paddle.Tensor,
                value: Union[float, int]):
    assert is_broadcastable(xs.shape, mask.shape) is True
    bshape = paddle.broadcast_shape(xs.shape, mask.shape)
    mask = mask.broadcast_to(bshape)
    trues = paddle.ones_like(xs) * value
    xs = paddle.where(mask, trues, xs)
    return xs


if not hasattr(paddle.Tensor, 'masked_fill'):
    logger.warn(
        "register user masked_fill to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.masked_fill = masked_fill


def masked_fill_(xs: paddle.Tensor,
                 mask: paddle.Tensor,
                 value: Union[float, int]) -> paddle.Tensor:
    assert is_broadcastable(xs.shape, mask.shape) is True
    bshape = paddle.broadcast_shape(xs.shape, mask.shape)
    mask = mask.broadcast_to(bshape)
    trues = paddle.ones_like(xs) * value
    ret = paddle.where(mask, trues, xs)
    paddle.assign(ret.detach(), output=xs)
    return xs


if not hasattr(paddle.Tensor, 'masked_fill_'):
    logger.warn(
        "register user masked_fill_ to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.masked_fill_ = masked_fill_


def fill_(xs: paddle.Tensor, value: Union[float, int]) -> paddle.Tensor:
    val = paddle.full_like(xs, value)
    paddle.assign(val.detach(), output=xs)
    return xs


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
    if isinstance(args[0], str):  # dtype
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


def func_int(x: paddle.Tensor) -> paddle.Tensor:
    return x.astype(paddle.int)


if not hasattr(paddle.Tensor, 'int'):
    logger.warn("register user int to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'int', func_int)


def tolist(x: paddle.Tensor) -> List[Any]:
    return x.numpy().tolist()


if not hasattr(paddle.Tensor, 'tolist'):
    logger.warn(
        "register user tolist to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'tolist', tolist)


########### hack paddle.nn #############
class GLU(nn.Layer):
    """Gated Linear Units (GLU) Layer"""

    def __init__(self, dim: int=-1):
        super().__init__()
        self.dim = dim

    def forward(self, xs):
        return F.glu(xs, axis=self.dim)


if not hasattr(paddle.nn, 'GLU'):
    logger.warn("register user GLU to paddle.nn, remove this when fixed!")
    setattr(paddle.nn, 'GLU', GLU)
