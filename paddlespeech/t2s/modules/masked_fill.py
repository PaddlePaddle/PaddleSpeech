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
from typing import Union

import paddle


def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


# assume that len(shp1) == len(shp2)
def broadcast_shape(shp1, shp2):
    result = []
    for a, b in zip(shp1[::-1], shp2[::-1]):
        is_a_int = isinstance(a, int)
        is_b_int = isinstance(b, int)

        if is_a_int and is_b_int:
            result.append(max(a, b))

        else:
            dtype = None
            if hasattr(a, 'dtype'):
                dtype = a.dtype
            if hasattr(b, 'dtype'):
                dtype = b.dtype

            if (is_a_int):
                a = paddle.full((), a, dtype=dtype)

            if (is_b_int):
                b = paddle.full((), b, dtype=dtype)

            result.append(paddle.maximum(a, b))

    return result[::-1]


def masked_fill(xs: paddle.Tensor,
                mask: paddle.Tensor,
                value: Union[float, int]):
    # comment following line for converting dygraph to static graph. 
    # assert is_broadcastable(xs.shape, mask.shape) is True
    bshape = broadcast_shape(xs.shape, mask.shape)
    mask.stop_gradient = True
    mask = mask.broadcast_to(bshape)
    trues = paddle.ones_like(xs) * value
    mask = mask.cast(dtype=paddle.bool)
    xs = paddle.where(mask, trues, xs)
    return xs
