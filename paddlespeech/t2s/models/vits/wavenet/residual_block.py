# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import math
from typing import Optional
from typing import Tuple

import paddle
import paddle.nn.functional as F
from paddle import nn


class ResidualBlock(nn.Layer):
    """Residual block module in WaveNet."""

    def __init__(
            self,
            kernel_size: int=3,
            residual_channels: int=64,
            gate_channels: int=128,
            skip_channels: int=64,
            aux_channels: int=80,
            global_channels: int=-1,
            dropout_rate: float=0.0,
            dilation: int=1,
            bias: bool=True,
            scale_residual: bool=False, ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int):
                Kernel size of dilation convolution layer.
            residual_channels (int):
                Number of channels for residual connection.
            skip_channels (int):
                Number of channels for skip connection.
            aux_channels (int):
                Number of local conditioning channels.
            dropout (float):
                Dropout probability.
            dilation (int):
                Dilation factor.
            bias (bool):
                Whether to add bias parameter in convolution layers.
            scale_residual (bool):
                Whether to scale the residual outputs.

        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.scale_residual = scale_residual

        # check
        assert (
            kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert gate_channels % 2 == 0

        # dilation conv
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1D(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias_attr=bias, )

        # local conditioning
        if aux_channels > 0:
            self.conv1x1_aux = nn.Conv1D(
                aux_channels, gate_channels, kernel_size=1, bias_attr=False)
        else:
            self.conv1x1_aux = None

        # global conditioning
        if global_channels > 0:
            self.conv1x1_glo = nn.Conv1D(
                global_channels, gate_channels, kernel_size=1, bias_attr=False)
        else:
            self.conv1x1_glo = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2

        # NOTE: concat two convs into a single conv for the efficiency
        #   (integrate res 1x1 + skip 1x1 convs)
        self.conv1x1_out = nn.Conv1D(
            gate_out_channels,
            residual_channels + skip_channels,
            kernel_size=1,
            bias_attr=bias)

    def forward(
            self,
            x: paddle.Tensor,
            x_mask: Optional[paddle.Tensor]=None,
            c: Optional[paddle.Tensor]=None,
            g: Optional[paddle.Tensor]=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            x_mask Optional[paddle.Tensor]: Mask tensor (B, 1, T).
            c (Optional[Tensor]): Local conditioning tensor (B, aux_channels, T).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = x
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv(x)

        # split into two part for gated activation
        splitdim = 1
        xa, xb = paddle.split(x, 2, axis=splitdim)

        # local conditioning
        if c is not None:
            c = self.conv1x1_aux(c)
            ca, cb = paddle.split(c, 2, axis=splitdim)
            xa, xb = xa + ca, xb + cb

        # global conditioning
        if g is not None:
            g = self.conv1x1_glo(g)
            ga, gb = paddle.split(g, 2, axis=splitdim)
            xa, xb = xa + ga, xb + gb

        x = paddle.tanh(xa) * F.sigmoid(xb)

        # residual + skip 1x1 conv
        x = self.conv1x1_out(x)
        if x_mask is not None:
            x = x * x_mask

        # split integrated conv results
        x, s = paddle.split(
            x, [self.residual_channels, self.skip_channels], axis=1)

        # for residual connection
        x = x + residual
        if self.scale_residual:
            x = x * math.sqrt(0.5)

        return x, s
