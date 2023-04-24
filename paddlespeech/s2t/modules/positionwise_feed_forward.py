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

from paddlespeech.s2t.modules.align import Linear
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ["PositionwiseFeedForward"]


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
