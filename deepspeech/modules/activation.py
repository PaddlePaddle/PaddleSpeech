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
import math

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

logger = logging.getLogger(__name__)

__all__ = ['brelu', "softplus", "gelu_accurate", "gelu", 'Swish']


def brelu(x, t_min=0.0, t_max=24.0, name=None):
    # paddle.to_tensor is dygraph_only can not work under JIT
    t_min = paddle.full(shape=[1], fill_value=t_min, dtype='float32')
    t_max = paddle.full(shape=[1], fill_value=t_max, dtype='float32')
    return x.maximum(t_min).minimum(t_max)


def softplus(x):
    """Softplus function."""
    if hasattr(paddle.nn.functional, 'softplus'):
        #return paddle.nn.functional.softplus(x.float()).type_as(x)
        return paddle.nn.functional.softplus(x)
    else:
        raise NotImplementedError


def gelu_accurate(x):
    """Gaussian Error Linear Units (GELU) activation."""
    # [reference] https://github.com/pytorch/fairseq/blob/e75cff5f2c1d62f12dc911e0bf420025eb1a4e33/fairseq/modules/gelu.py
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + paddle.tanh(gelu_accurate._a *
                                      (x + 0.044715 * paddle.pow(x, 3))))


def gelu(x):
    """Gaussian Error Linear Units (GELU) activation."""
    if hasattr(torch.nn.functional, 'gelu'):
        #return torch.nn.functional.gelu(x.float()).type_as(x)
        return torch.nn.functional.gelu(x)
    else:
        return x * 0.5 * (1.0 + paddle.erf(x / math.sqrt(2.0)))


class Swish(nn.Layer):
    """Construct an Swish object."""

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Return Swish activation function."""
        return x * F.sigmoid(x)
