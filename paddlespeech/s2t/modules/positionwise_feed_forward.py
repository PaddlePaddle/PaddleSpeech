# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Mobvoi Inc. All Rights Reserved.
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
# Modified from wenet(https://github.com/wenet-e2e/wenet)
"""Positionwise feed forward layer definition."""
import paddle
from paddle import nn
from paddle.nn import initializer as I

from paddlespeech.s2t.modules.align import Linear
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ["PositionwiseFeedForward", "PositionwiseFeedForward2"]


class PositionwiseFeedForward(nn.Layer):
    """Positionwise feed forward layer."""

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: nn.Layer=nn.ReLU()):
        """Construct a PositionwiseFeedForward object.

        FeedForward are appied on each position of the sequence.
        The output dim is same with the input dim.

        Args:
            idim (int): Input dimenstion.
            hidden_units (int): The number of hidden units.
            dropout_rate (float): Dropout rate.
            activation (paddle.nn.Layer): Activation function
        """
        super().__init__()
        self.w_1 = Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = Linear(hidden_units, idim)

    def forward(self, xs: paddle.Tensor) -> paddle.Tensor:
        """Forward function.
        Args:
            xs: input tensor (B, Lmax, D)
        Returns:
            output tensor, (B, Lmax, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class PositionwiseFeedForward2(paddle.nn.Layer):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (paddle.nn.Layer): Activation function
    """

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: paddle.nn.Layer=paddle.nn.ReLU(),
                 adaptive_scale: bool=False,
                 init_weights: bool=False):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward2, self).__init__()
        self.idim = idim
        self.hidden_units = hidden_units
        self.w_1 = Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = paddle.nn.Dropout(dropout_rate)
        self.w_2 = Linear(hidden_units, idim)
        self.adaptive_scale = adaptive_scale
        ada_scale = self.create_parameter(
            [1, 1, idim], default_initializer=I.XavierUniform())
        self.add_parameter('ada_scale', ada_scale)
        ada_bias = self.create_parameter(
            [1, 1, idim], default_initializer=I.XavierUniform())
        self.add_parameter('ada_bias', ada_bias)

        if init_weights:
            self.init_weights()

    def init_weights(self):
        ffn1_max = self.idim**-0.5
        ffn2_max = self.hidden_units**-0.5
        self.w_1._param_attr = paddle.nn.initializer.Uniform(
            low=-ffn1_max, high=ffn1_max)
        self.w_1._bias_attr = paddle.nn.initializer.Uniform(
            low=-ffn1_max, high=ffn1_max)
        self.w_2._param_attr = paddle.nn.initializer.Uniform(
            low=-ffn2_max, high=ffn2_max)
        self.w_2._bias_attr = paddle.nn.initializer.Uniform(
            low=-ffn2_max, high=ffn2_max)

    def forward(self, xs: paddle.Tensor) -> paddle.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        if self.adaptive_scale:
            xs = self.ada_scale * xs + self.ada_bias
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))
