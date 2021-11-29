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
# Modified from espnet(https://github.com/espnet/espnet)
"""Positionwise feed forward layer definition."""
import paddle
from paddle import nn


class PositionwiseFeedForward(nn.Layer):
    """Positionwise feed forward layer.

    Parameters
    ----------
    idim : int
        Input dimenstion.
    hidden_units : int
        The number of hidden units.
    dropout_rate : float
        Dropout rate.
    """

    def __init__(self,
                 idim,
                 hidden_units,
                 dropout_rate,
                 activation=paddle.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super().__init__()
        self.w_1 = paddle.nn.Linear(idim, hidden_units, bias_attr=True)
        self.w_2 = paddle.nn.Linear(hidden_units, idim, bias_attr=True)
        self.dropout = paddle.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward funciton."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
