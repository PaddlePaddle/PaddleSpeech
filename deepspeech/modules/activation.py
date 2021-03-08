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
import numpy as np

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

logger = logging.getLogger(__name__)

__all__ = ['brelu']


def brelu(x, t_min=0.0, t_max=24.0, name=None):
    # paddle.to_tensor is dygraph_only can not work under JIT
    t_min = paddle.full(shape=[1], fill_value=t_min, dtype='float32')
    t_max = paddle.full(shape=[1], fill_value=t_max, dtype='float32')
    return x.maximum(t_min).minimum(t_max)
