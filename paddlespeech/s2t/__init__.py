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

if not hasattr(paddle, 'softmax'):
    logger.debug("register user softmax to paddle, remove this when fixed!")
    setattr(paddle, 'softmax', paddle.nn.functional.softmax)

if not hasattr(paddle, 'log_softmax'):
    logger.debug("register user log_softmax to paddle, remove this when fixed!")
    setattr(paddle, 'log_softmax', paddle.nn.functional.log_softmax)

if not hasattr(paddle, 'sigmoid'):
    logger.debug("register user sigmoid to paddle, remove this when fixed!")
    setattr(paddle, 'sigmoid', paddle.nn.functional.sigmoid)

if not hasattr(paddle, 'log_sigmoid'):
    logger.debug("register user log_sigmoid to paddle, remove this when fixed!")
    setattr(paddle, 'log_sigmoid', paddle.nn.functional.log_sigmoid)

if not hasattr(paddle, 'relu'):
    logger.debug("register user relu to paddle, remove this when fixed!")
    setattr(paddle, 'relu', paddle.nn.functional.relu)


def cat(xs, dim=0):
    return paddle.concat(xs, axis=dim)


if not hasattr(paddle, 'cat'):
    logger.debug(
        "override cat of paddle if exists or register, remove this when fixed!")
    paddle.cat = cat


########### hack paddle.Tensor #############
def item(x: paddle.Tensor):
    return x.numpy().item()


if not hasattr(paddle.Tensor, 'item'):
    logger.debug(
        "override item of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.item = item


def func_long(x: paddle.Tensor):
    return paddle.cast(x, paddle.long)


if not hasattr(paddle.Tensor, 'long'):
    logger.debug(
        "override long of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.long = func_long
    paddle.static.Variable.long = func_long

if not hasattr(paddle.Tensor, 'numel'):
    logger.debug(
        "override numel of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.numel = paddle.numel
    paddle.static.Variable.numel = paddle.numel


def new_full(x: paddle.Tensor,
             size: Union[List[int], Tuple[int], paddle.Tensor],
             fill_value: Union[float, int, bool, paddle.Tensor],
             dtype=None):
    return paddle.full(size, fill_value, dtype=x.dtype)


if not hasattr(paddle.Tensor, 'new_full'):
    logger.debug(
        "override new_full of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.new_full = new_full
    paddle.static.Variable.new_full = new_full

def contiguous(xs: paddle.Tensor) -> paddle.Tensor:
    return xs


if not hasattr(paddle.Tensor, 'contiguous'):
    logger.debug(
        "override contiguous of paddle.Tensor if exists or register, remove this when fixed!"
    )
    paddle.Tensor.contiguous = contiguous
    paddle.static.Variable.contiguous = contiguous


def view(xs: paddle.Tensor, *args: int) -> paddle.Tensor:
    return xs.reshape(args)


if not hasattr(paddle.Tensor, 'view'):
    logger.debug("register user view to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.view = view
    paddle.static.Variable.view = view


def view_as(xs: paddle.Tensor, ys: paddle.Tensor) -> paddle.Tensor:
    return xs.reshape(paddle.shape(ys))


if not hasattr(paddle.Tensor, 'view_as'):
    logger.debug(
        "register user view_as to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.view_as = view_as
    paddle.static.Variable.view_as = view_as


def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def broadcast_shape(shp1, shp2):
    result = []
    for a, b in zip(shp1[::-1], shp2[::-1]):
        result.append(max(a, b))
    return result[::-1]


def masked_fill(xs: paddle.Tensor,
                mask: paddle.Tensor,
                value: Union[float, int]):
    bshape = broadcast_shape(xs.shape, mask.shape)
    mask.stop_gradient = True
    tmp = paddle.ones(shape=[len(bshape)], dtype='int32')
    for index in range(len(bshape)):
        tmp[index] = bshape[index]
    mask = mask.broadcast_to(tmp)
    trues = paddle.ones_like(xs) * value
    xs = paddle.where(mask, trues, xs)
    return xs


if not hasattr(paddle.Tensor, 'masked_fill'):
    logger.debug(
        "register user masked_fill to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.masked_fill = masked_fill
    paddle.static.Variable.masked_fill = masked_fill


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
    logger.debug(
        "register user masked_fill_ to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.masked_fill_ = masked_fill_
    paddle.static.Variable.maksed_fill_ = masked_fill_


def fill_(xs: paddle.Tensor, value: Union[float, int]) -> paddle.Tensor:
    val = paddle.full_like(xs, value)
    paddle.assign(val.detach(), output=xs)
    return xs


if not hasattr(paddle.Tensor, 'fill_'):
    logger.debug(
        "register user fill_ to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.fill_ = fill_
    paddle.static.Variable.fill_ = fill_


def repeat(xs: paddle.Tensor, *size: Any) -> paddle.Tensor:
    return paddle.tile(xs, size)


if not hasattr(paddle.Tensor, 'repeat'):
    logger.debug(
        "register user repeat to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.repeat = repeat
    paddle.static.Variable.repeat = repeat

if not hasattr(paddle.Tensor, 'softmax'):
    logger.debug(
        "register user softmax to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'softmax', paddle.nn.functional.softmax)

if not hasattr(paddle.Tensor, 'sigmoid'):
    logger.debug(
        "register user sigmoid to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'sigmoid', paddle.nn.functional.sigmoid)

if not hasattr(paddle.Tensor, 'relu'):
    logger.debug("register user relu to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'relu', paddle.nn.functional.relu)


def type_as(x: paddle.Tensor, other: paddle.Tensor) -> paddle.Tensor:
    return x.astype(other.dtype)


if not hasattr(paddle.Tensor, 'type_as'):
    logger.debug(
        "register user type_as to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'type_as', type_as)
    setattr(paddle.static.Variable, 'type_as', type_as)


def to(x: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
    assert len(args) == 1
    if isinstance(args[0], str):  # dtype
        return x.astype(args[0])
    elif isinstance(args[0], paddle.Tensor):  # Tensor
        return x.astype(args[0].dtype)
    else:  # Device
        return x


if not hasattr(paddle.Tensor, 'to'):
    logger.debug("register user to to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'to', to)
    setattr(paddle.static.Variable, 'to', to)


def func_float(x: paddle.Tensor) -> paddle.Tensor:
    return x.astype(paddle.float)


if not hasattr(paddle.Tensor, 'float'):
    logger.debug(
        "register user float to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'float', func_float)
    setattr(paddle.static.Variable, 'float', func_float)


def func_int(x: paddle.Tensor) -> paddle.Tensor:
    return x.astype(paddle.int)


if not hasattr(paddle.Tensor, 'int'):
    logger.debug("register user int to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'int', func_int)
    setattr(paddle.static.Variable, 'int', func_int)


def tolist(x: paddle.Tensor) -> List[Any]:
    return x.numpy().tolist()


if not hasattr(paddle.Tensor, 'tolist'):
    logger.debug(
        "register user tolist to paddle.Tensor, remove this when fixed!")
    setattr(paddle.Tensor, 'tolist', tolist)
    setattr(paddle.static.Variable, 'tolist', tolist)

########### hack paddle.nn #############
from paddle.nn import Layer
from typing import Optional
from typing import Mapping
from typing import Iterable
from typing import Tuple
from typing import Iterator
from collections import OrderedDict, abc as container_abcs


class LayerDict(paddle.nn.Layer):
    r"""Holds submodules in a dictionary.

    :class:`~paddle.nn.LayerDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~paddle.nn.Layer` methods.

    :class:`~paddle.nn.LayerDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~paddle.nn.LayerDict.update`, the order of the merged
      ``OrderedDict``, ``dict`` (started from Python 3.6) or another
      :class:`~paddle.nn.LayerDict` (the argument to
      :meth:`~paddle.nn.LayerDict.update`).

    Note that :meth:`~paddle.nn.LayerDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict`` before Python version 3.6) does not
    preserve the order of the merged mapping.

    Args:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Layer):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.LayerDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.LayerDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules: Optional[Mapping[str, Layer]]=None) -> None:
        super(LayerDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Layer:
        return self._modules[key]

    def __setitem__(self, key: str, module: Layer) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the LayerDict.
        """
        self._modules.clear()

    def pop(self, key: str) -> Layer:
        r"""Remove key from the LayerDict and return its module.

        Args:
            key (string): key to pop from the LayerDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the LayerDict keys.
        """
        return self._modules.keys()

    def items(self) -> Iterable[Tuple[str, Layer]]:
        r"""Return an iterable of the LayerDict key/value pairs.
        """
        return self._modules.items()

    def values(self) -> Iterable[Layer]:
        r"""Return an iterable of the LayerDict values.
        """
        return self._modules.values()

    def update(self, modules: Mapping[str, Layer]) -> None:
        r"""Update the :class:`~paddle.nn.LayerDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~paddle.nn.LayerDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Args:
            modules (iterable): a mapping (dictionary) from string to :class:`~paddle.nn.Layer`,
                or an iterable of key-value pairs of type (string, :class:`~paddle.nn.Layer`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("LayerDict.update should be called with an "
                            "iterable of key/value pairs, but got " + type(
                                modules).__name__)

        if isinstance(modules,
                      (OrderedDict, LayerDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            # modules here can be a list with two items
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError("LayerDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("LayerDict update sequence element "
                                     "#" + str(j) + " has length " + str(
                                         len(m)) + "; 2 is required")
                # modules can be Mapping (what it's typed at), or a list: [(name1, module1), (name2, module2)]
                # that's too cumbersome to type correctly with overloads, so we add an ignore here
                self[m[0]] = m[1]  # type: ignore[assignment]

    # remove forward alltogether to fallback on Module's _forward_unimplemented


if not hasattr(paddle.nn, 'LayerDict'):
    logger.debug(
        "register user LayerDict to paddle.nn, remove this when fixed!")
    setattr(paddle.nn, 'LayerDict', LayerDict)
