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
import paddle.nn.functional as F
from paddle import nn


class GLU(nn.Layer):
    """Gated Linear Units (GLU) Layer"""

    def __init__(self, dim: int=-1):
        super().__init__()
        self.dim = dim

    def forward(self, xs):
        return F.glu(xs, axis=self.dim)


def get_activation(act):
    """Return activation function."""

    activation_funcs = {
        "hardtanh": paddle.nn.Hardtanh,
        "tanh": paddle.nn.Tanh,
        "relu": paddle.nn.ReLU,
        "selu": paddle.nn.SELU,
        "swish": paddle.nn.Swish,
        "glu": GLU
    }

    return activation_funcs[act]()
