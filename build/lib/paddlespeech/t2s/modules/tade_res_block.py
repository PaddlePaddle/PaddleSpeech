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
"""StyleMelGAN's TADEResBlock Modules."""
from functools import partial

import paddle.nn.functional as F
from paddle import nn


class TADELayer(nn.Layer):
    """TADE Layer module."""

    def __init__(
            self,
            in_channels: int=64,
            aux_channels: int=80,
            kernel_size: int=9,
            bias: bool=True,
            upsample_factor: int=2,
            upsample_mode: str="nearest", ):
        """Initilize TADE layer."""
        super().__init__()
        self.norm = nn.InstanceNorm1D(
            in_channels,
            momentum=0.1,
            data_format="NCL",
            weight_attr=False,
            bias_attr=False)
        self.aux_conv = nn.Sequential(
            nn.Conv1D(
                aux_channels,
                in_channels,
                kernel_size,
                1,
                bias_attr=bias,
                padding=(kernel_size - 1) // 2, ), )
        self.gated_conv = nn.Sequential(
            nn.Conv1D(
                in_channels,
                in_channels * 2,
                kernel_size,
                1,
                bias_attr=bias,
                padding=(kernel_size - 1) // 2, ), )
        self.upsample = nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode)

    def forward(self, x, c):
        """Calculate forward propagation.
        Args:
            x (Tensor): 
                Input tensor (B, in_channels, T).
            c (Tensor): 
                Auxiliary input tensor (B, aux_channels, T).
        Returns:
            Tensor: 
                Output tensor (B, in_channels, T * upsample_factor).
            Tensor:
                Upsampled aux tensor (B, in_channels, T * upsample_factor).
        """

        x = self.norm(x)
        # 'bilinear', 'bicubic' and 'nearest' only support 4-D tensor.
        c = self.upsample(c.unsqueeze(-1))
        c = c[:, :, :, 0]

        c = self.aux_conv(c)
        cg = self.gated_conv(c)
        cg1, cg2 = cg.split(2, axis=1)
        # 'bilinear', 'bicubic' and 'nearest' only support 4-D tensor.
        y = cg1 * self.upsample(x.unsqueeze(-1))[:, :, :, 0] + cg2
        return y, c


class TADEResBlock(nn.Layer):
    """TADEResBlock module."""

    def __init__(
            self,
            in_channels: int=64,
            aux_channels: int=80,
            kernel_size: int=9,
            dilation: int=2,
            bias: bool=True,
            upsample_factor: int=2,
            # this is a diff in paddle, the mode only can be "linear" when input is 3D
            upsample_mode: str="nearest",
            gated_function: str="softmax", ):
        """Initialize TADEResBlock module."""
        super().__init__()
        self.tade1 = TADELayer(
            in_channels=in_channels,
            aux_channels=aux_channels,
            kernel_size=kernel_size,
            bias=bias,
            upsample_factor=1,
            upsample_mode=upsample_mode, )
        self.gated_conv1 = nn.Conv1D(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias_attr=bias,
            padding=(kernel_size - 1) // 2, )
        self.tade2 = TADELayer(
            in_channels=in_channels,
            aux_channels=in_channels,
            kernel_size=kernel_size,
            bias=bias,
            upsample_factor=upsample_factor,
            upsample_mode=upsample_mode, )
        self.gated_conv2 = nn.Conv1D(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias_attr=bias,
            dilation=dilation,
            padding=(kernel_size - 1) // 2 * dilation, )
        self.upsample = nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode)
        if gated_function == "softmax":
            self.gated_function = partial(F.softmax, axis=1)
        elif gated_function == "sigmoid":
            self.gated_function = F.sigmoid
        else:
            raise ValueError(f"{gated_function} is not supported.")

    def forward(self, x, c):
        """Calculate forward propagation.
        Args:

            x (Tensor): 
                Input tensor (B, in_channels, T).
            c (Tensor): 
                Auxiliary input tensor (B, aux_channels, T).
        Returns:
            Tensor: 
                Output tensor (B, in_channels, T * upsample_factor).
            Tensor: 
                Upsampled auxirialy tensor (B, in_channels, T * upsample_factor).
        """
        residual = x
        x, c = self.tade1(x, c)
        x = self.gated_conv1(x)
        xa, xb = x.split(2, axis=1)
        x = self.gated_function(xa) * F.tanh(xb)
        x, c = self.tade2(x, c)
        x = self.gated_conv2(x)
        xa, xb = x.split(2, axis=1)
        x = self.gated_function(xa) * F.tanh(xb)
        # 'bilinear', 'bicubic' and 'nearest' only support 4-D tensor.
        return self.upsample(residual.unsqueeze(-1))[:, :, :, 0] + x, c
