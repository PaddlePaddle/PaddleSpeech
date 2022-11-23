# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import Tensor

__all__ = ["sinusoid_position_encoding", "scaled_position_encoding"]


def sinusoid_position_encoding(num_positions: int,
                               feature_size: int,
                               omega: float=1.0,
                               start_pos: int=0,
                               dtype=None) -> paddle.Tensor:
    # return tensor shape (num_positions, feature_size)
    # NOTE: to be compatible with paddle's to_static, we cannnot raise 
    # an exception here, take care of it by yourself
    # if (feature_size % 2 != 0):
    #     raise ValueError("size should be divisible by 2")
    dtype = dtype or paddle.get_default_dtype()

    channel = paddle.arange(0, feature_size, 2, dtype=dtype)
    index = paddle.arange(start_pos, start_pos + num_positions, 1, dtype=dtype)
    denominator = channel / float(feature_size)
    denominator = paddle.to_tensor([10000.0], dtype='float32')**denominator
    p = (paddle.unsqueeze(index, -1) * omega) / denominator
    encodings = paddle.zeros([num_positions, feature_size], dtype=dtype)
    encodings[:, 0::2] = paddle.sin(p)
    encodings[:, 1::2] = paddle.cos(p)
    return encodings


def scaled_position_encoding(num_positions: int,
                             feature_size: int,
                             omega: Tensor,
                             start_pos: int=0,
                             dtype=None) -> Tensor:
    # omega: Tensor (batch_size, )
    # return tensor shape (batch_size, num_positions, feature_size)
    # consider renaming this as batched positioning encoding
    if (feature_size % 2 != 0):
        raise ValueError("size should be divisible by 2")
    dtype = dtype or paddle.get_default_dtype()

    channel = paddle.arange(0, feature_size, 2, dtype=dtype)
    index = paddle.arange(
        start_pos, start_pos + num_positions, 1, dtype=omega.dtype)
    batch_size = omega.shape[0]
    omega = paddle.unsqueeze(omega, [1, 2])
    p = (paddle.unsqueeze(index, -1) *
         omega) / (10000.0**(channel / float(feature_size)))
    encodings = paddle.zeros(
        [batch_size, num_positions, feature_size], dtype=dtype)
    # it is nice to have fancy indexing and inplace operations
    encodings[:, :, 0::2] = paddle.sin(p)
    encodings[:, :, 1::2] = paddle.cos(p)
    return encodings
