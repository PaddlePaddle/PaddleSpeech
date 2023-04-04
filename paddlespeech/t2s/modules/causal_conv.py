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
"""Causal convolusion layer modules."""
import paddle
from paddle import nn


class CausalConv1D(nn.Layer):
    """CausalConv1D module with customized initialization."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        bias=True,
        pad="Pad1D",
        pad_params={"value": 0.0},
    ):
        """Initialize CausalConv1d module."""
        super().__init__()
        self.pad = getattr(paddle.nn, pad)((kernel_size - 1) * dilation,
                                           **pad_params)
        self.conv = nn.Conv1D(in_channels,
                              out_channels,
                              kernel_size,
                              dilation=dilation,
                              bias_attr=bias)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): 
                Input tensor (B, in_channels, T).
        Returns: 
            Tensor: Output tensor (B, out_channels, T).
        """
        return self.conv(self.pad(x))[:, :, :x.shape[2]]


class CausalConv1DTranspose(nn.Layer):
    """CausalConv1DTranspose module with customized initialization."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bias=True):
        """Initialize CausalConvTranspose1d module."""
        super().__init__()
        self.deconv = nn.Conv1DTranspose(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         bias_attr=bias)
        self.stride = stride

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): 
                Input tensor (B, in_channels, T_in).
        Returns:
            Tensor: Output tensor (B, out_channels, T_out).
        """
        return self.deconv(x)[:, :, :-self.stride]
