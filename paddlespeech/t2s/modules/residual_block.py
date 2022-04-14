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
import math
from typing import Any
from typing import Dict
from typing import List

import paddle
from paddle import nn
from paddle.nn import functional as F

from paddlespeech.t2s.modules.activation import get_activation


class WaveNetResidualBlock(nn.Layer):
    """A gated activation unit composed of an 1D convolution, a gated tanh
    unit and parametric redidual and skip connections. For more details, 
    refer to `WaveNet: A Generative Model for Raw Audio <https://arxiv.org/abs/1609.03499>`_.

    Args:
        kernel_size (int, optional): Kernel size of the 1D convolution, by default 3
        residual_channels (int, optional): Feature size of the residual output(and also the input), by default 64
        gate_channels (int, optional): Output feature size of the 1D convolution, by default 128
        skip_channels (int, optional): Feature size of the skip output, by default 64
        aux_channels (int, optional): Feature size of the auxiliary input (e.g. spectrogram), by default 80
        dropout (float, optional): Probability of the dropout before the 1D convolution, by default 0.
        dilation (int, optional): Dilation of the 1D convolution, by default 1
        bias (bool, optional): Whether to use bias in the 1D convolution, by default True
        use_causal_conv (bool, optional): Whether to use causal padding for the 1D convolution, by default False
    """

    def __init__(self,
                 kernel_size: int=3,
                 residual_channels: int=64,
                 gate_channels: int=128,
                 skip_channels: int=64,
                 aux_channels: int=80,
                 dropout: float=0.,
                 dilation: int=1,
                 bias: bool=True,
                 use_causal_conv: bool=False):
        super().__init__()
        self.dropout = dropout
        if use_causal_conv:
            padding = (kernel_size - 1) * dilation
        else:
            assert kernel_size % 2 == 1
            padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv

        self.conv = nn.Conv1D(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias_attr=bias)
        if aux_channels is not None:
            self.conv1x1_aux = nn.Conv1D(
                aux_channels, gate_channels, kernel_size=1, bias_attr=False)
        else:
            self.conv1x1_aux = None

        gate_out_channels = gate_channels // 2
        self.conv1x1_out = nn.Conv1D(
            gate_out_channels, residual_channels, kernel_size=1, bias_attr=bias)
        self.conv1x1_skip = nn.Conv1D(
            gate_out_channels, skip_channels, kernel_size=1, bias_attr=bias)

    def forward(self, x, c):
        """
        Args:
            x (Tensor): the input features. Shape (N, C_res, T)
            c (Tensor): the auxiliary input. Shape (N, C_aux, T)

        Returns:
            res (Tensor): Shape (N, C_res, T), the residual output, which is used as the 
                input of the next ResidualBlock in a stack of ResidualBlocks.
            skip (Tensor): Shape (N, C_skip, T), the skip output, which is collected among
                each layer in a stack of ResidualBlocks.
        """
        x_input = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv(x)
        x = x[:, :, x_input.shape[-1]] if self.use_causal_conv else x
        if c is not None:
            c = self.conv1x1_aux(c)
            x += c

        a, b = paddle.chunk(x, 2, axis=1)
        x = paddle.tanh(a) * F.sigmoid(b)

        skip = self.conv1x1_skip(x)
        res = (self.conv1x1_out(x) + x_input) * math.sqrt(0.5)
        return res, skip


class HiFiGANResidualBlock(nn.Layer):
    """Residual block module in HiFiGAN."""

    def __init__(
            self,
            kernel_size: int=3,
            channels: int=512,
            dilations: List[int]=(1, 3, 5),
            bias: bool=True,
            use_additional_convs: bool=True,
            nonlinear_activation: str="leakyrelu",
            nonlinear_activation_params: Dict[str, Any]={"negative_slope": 0.1},
    ):
        """Initialize HiFiGANResidualBlock module.
        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            dilations (List[int]): List of dilation factors.
            use_additional_convs (bool): Whether to use additional convolution layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
        """
        super().__init__()

        self.use_additional_convs = use_additional_convs
        self.convs1 = nn.LayerList()
        if use_additional_convs:
            self.convs2 = nn.LayerList()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."

        for dilation in dilations:
            self.convs1.append(
                nn.Sequential(
                    get_activation(nonlinear_activation, **
                                   nonlinear_activation_params),
                    nn.Conv1D(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        bias_attr=bias,
                        padding=(kernel_size - 1) // 2 * dilation, ), ))
            if use_additional_convs:
                self.convs2.append(
                    nn.Sequential(
                        get_activation(nonlinear_activation, **
                                       nonlinear_activation_params),
                        nn.Conv1D(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            bias_attr=bias,
                            padding=(kernel_size - 1) // 2, ), ))

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, channels, T).
        Returns:
            Tensor: Output tensor (B, channels, T).
        """
        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x
