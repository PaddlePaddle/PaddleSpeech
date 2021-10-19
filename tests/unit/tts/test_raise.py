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
import paddle
from paddle.static import InputSpec


def sinusoid_position_encoding(num_positions: int,
                               feature_size: int,
                               omega: float=1.0,
                               start_pos: int=0,
                               dtype=None) -> paddle.Tensor:
    # return tensor shape (num_positions, feature_size)

    if (feature_size % 2 != 0):
        raise ValueError("size should be divisible by 2")
    dtype = dtype or paddle.get_default_dtype()

    channel = paddle.arange(0, feature_size, 2, dtype=dtype)
    index = paddle.arange(start_pos, start_pos + num_positions, 1, dtype=dtype)
    p = (paddle.unsqueeze(index, -1) *
         omega) / (10000.0**(channel / float(feature_size)))
    encodings = paddle.zeros([num_positions, feature_size], dtype=dtype)
    encodings[:, 0::2] = paddle.sin(p)
    encodings[:, 1::2] = paddle.cos(p)
    return encodings


def call_it(x):
    shape = paddle.shape(x)
    a = shape[0]
    b = shape[1]
    c = sinusoid_position_encoding(a, b)
    return c


call_it(paddle.randn([8, 32]))
m = paddle.jit.to_static(
    call_it, input_spec=[InputSpec([-1, -1], dtype=paddle.int32)])
m(paddle.randn([8, 32]).astype(paddle.int32))
