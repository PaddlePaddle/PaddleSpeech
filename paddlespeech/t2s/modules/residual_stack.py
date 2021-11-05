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
"""Residual stack module in MelGAN."""
from typing import Any
from typing import Dict

from paddle import nn

from paddlespeech.t2s.modules.causal_conv import CausalConv1D


class ResidualStack(nn.Layer):
    """Residual stack module introduced in MelGAN."""

    def __init__(
            self,
            kernel_size: int=3,
            channels: int=32,
            dilation: int=1,
            bias: bool=True,
            nonlinear_activation: str="LeakyReLU",
            nonlinear_activation_params: Dict[str, Any]={"negative_slope": 0.2},
            pad: str="Pad1D",
            pad_params: Dict[str, Any]={"mode": "reflect"},
            use_causal_conv: bool=False, ):
        """Initialize ResidualStack module.
        Parameters
        ----------
        kernel_size : int
            Kernel size of dilation convolution layer.
        channels : int
            Number of channels of convolution layers.
        dilation : int
            Dilation factor.
        bias : bool
            Whether to add bias parameter in convolution layers.
        nonlinear_activation : str
            Activation function module name.
        nonlinear_activation_params : Dict[str,Any]
            Hyperparameters for activation function.
        pad : str
            Padding function module name before dilated convolution layer.
        pad_params : Dict[str, Any]
            Hyperparameters for padding function.
        use_causal_conv : bool
            Whether to use causal convolution.
        """
        super().__init__()

        # defile residual stack part
        if not use_causal_conv:
            assert (kernel_size - 1
                    ) % 2 == 0, "Not support even number kernel size."
            self.stack = nn.Sequential(
                getattr(nn, nonlinear_activation)(
                    **nonlinear_activation_params),
                getattr(nn, pad)((kernel_size - 1) // 2 * dilation,
                                 **pad_params),
                nn.Conv1D(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    bias_attr=bias),
                getattr(nn, nonlinear_activation)(
                    **nonlinear_activation_params),
                nn.Conv1D(channels, channels, 1, bias_attr=bias), )
        else:
            self.stack = nn.Sequential(
                getattr(nn, nonlinear_activation)(
                    **nonlinear_activation_params),
                CausalConv1D(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params, ),
                getattr(nn, nonlinear_activation)(
                    **nonlinear_activation_params),
                nn.Conv1D(channels, channels, 1, bias_attr=bias), )

        # defile extra layer for skip connection
        self.skip_layer = nn.Conv1D(channels, channels, 1, bias_attr=bias)

    def forward(self, c):
        """Calculate forward propagation.
        Parameters
        ----------
        c : Tensor
            Input tensor (B, channels, T).
        Returns
        ----------
        Tensor
            Output tensor (B, chennels, T).
        """
        return self.stack(c) + self.skip_layer(c)
