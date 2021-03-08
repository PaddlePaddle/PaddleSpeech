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

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

logger = logging.getLogger(__name__)

__all__ = ['sequence_mask']


def sequence_mask(x_len, max_len=None, dtype='float32'):
    max_len = max_len or x_len.max()
    x_len = paddle.unsqueeze(x_len, -1)
    row_vector = paddle.arange(max_len)
    #mask = row_vector < x_len
    mask = row_vector > x_len  # a bug, broadcast 的时候出错了
    mask = paddle.cast(mask, dtype)
    return mask
