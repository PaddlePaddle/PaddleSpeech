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
"""Basic Flow modules used in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""
import math
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
from paddle import nn

from paddlespeech.t2s.models.vits.transform import piecewise_rational_quadratic_transform


class FlipFlow(nn.Layer):
    """Flip flow module."""

    def forward(self, x: paddle.Tensor, *args, inverse: bool=False, **kwargs
                ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        """Calculate forward propagation.
        Args:
            x (Tensor):
                Input tensor (B, channels, T).
            inverse (bool):
                Whether to inverse the flow.
        Returns:
            Tensor:
                Flipped tensor (B, channels, T).
            Tensor:
                Log-determinant tensor for NLL (B,) if not inverse.
        """
        x = paddle.flip(x, [1])
        if not inverse:
            logdet = paddle.zeros(paddle.shape(x)[0:1], dtype=x.dtype)
            return x, logdet
        else:
            return x


class LogFlow(nn.Layer):
    """Log flow module."""

    def forward(self,
                x: paddle.Tensor,
                x_mask: paddle.Tensor,
                inverse: bool=False,
                eps: float=1e-5,
                **kwargs
                ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        """Calculate forward propagation.
        Args:
            x (Tensor):
                Input tensor (B, channels, T).
            x_mask (Tensor):
                Mask tensor (B, 1, T).
            inverse (bool):
                Whether to inverse the flow.
            eps (float):
                Epsilon for log.
        Returns:
            Tensor:
                Output tensor (B, channels, T).
            Tensor:
                Log-determinant tensor for NLL (B,) if not inverse.
        """
        if not inverse:
            y = paddle.log(paddle.clip(x, min=eps)) * x_mask
            logdet = paddle.sum(-y, [1, 2])
            return y, logdet
        else:
            x = paddle.exp(x) * x_mask
            return x


class ElementwiseAffineFlow(nn.Layer):
    """Elementwise affine flow module."""

    def __init__(self, channels: int):
        """Initialize ElementwiseAffineFlow module.
        Args:
            channels (int):
                Number of channels.
        """
        super().__init__()
        self.channels = channels

        m = paddle.zeros([channels, 1])
        self.m = paddle.create_parameter(
            shape=m.shape,
            dtype=str(m.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(m))
        logs = paddle.zeros([channels, 1])
        self.logs = paddle.create_parameter(
            shape=logs.shape,
            dtype=str(logs.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(logs))

    def forward(self,
                x: paddle.Tensor,
                x_mask: paddle.Tensor,
                inverse: bool=False,
                **kwargs
                ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        """Calculate forward propagation.
        Args:
            x (Tensor):
                Input tensor (B, channels, T).
            x_mask (Tensor):
                Mask tensor (B, 1, T).
            inverse (bool):
                Whether to inverse the flow.
        Returns:
            Tensor:
                Output tensor (B, channels, T).
            Tensor:
                Log-determinant tensor for NLL (B,) if not inverse.
        """
        if not inverse:
            y = self.m + paddle.exp(self.logs) * x
            y = y * x_mask
            logdet = paddle.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * paddle.exp(-self.logs) * x_mask
            return x


class Transpose(nn.Layer):
    """Transpose module for paddle.nn.Sequential()."""

    def __init__(self, dim1: int, dim2: int):
        """Initialize Transpose module."""
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Transpose."""
        len_dim = len(x.shape)
        orig_perm = list(range(len_dim))
        new_perm = orig_perm[:]
        temp = new_perm[self.dim1]
        new_perm[self.dim1] = new_perm[self.dim2]
        new_perm[self.dim2] = temp

        return paddle.transpose(x, new_perm)


class DilatedDepthSeparableConv(nn.Layer):
    """Dilated depth-separable conv module."""

    def __init__(
            self,
            channels: int,
            kernel_size: int,
            layers: int,
            dropout_rate: float=0.0,
            eps: float=1e-5, ):
        """Initialize DilatedDepthSeparableConv module.
        Args:
            channels (int):
                Number of channels.
            kernel_size (int):
                Kernel size.
            layers (int):
                Number of layers.
            dropout_rate (float):
                Dropout rate.
            eps (float):
                Epsilon for layer norm.
        """
        super().__init__()

        self.convs = nn.LayerList()
        for i in range(layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs.append(
                nn.Sequential(
                    nn.Conv1D(
                        channels,
                        channels,
                        kernel_size,
                        groups=channels,
                        dilation=dilation,
                        padding=padding, ),
                    Transpose(1, 2),
                    nn.LayerNorm(channels, epsilon=eps),
                    Transpose(1, 2),
                    nn.GELU(),
                    nn.Conv1D(
                        channels,
                        channels,
                        1, ),
                    Transpose(1, 2),
                    nn.LayerNorm(channels, epsilon=eps),
                    Transpose(1, 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate), ))

    def forward(self,
                x: paddle.Tensor,
                x_mask: paddle.Tensor,
                g: Optional[paddle.Tensor]=None) -> paddle.Tensor:
        """Calculate forward propagation.
        Args:
            x (Tensor):
                Input tensor (B, in_channels, T).
            x_mask (Tensor):
                Mask tensor (B, 1, T).
            g (Optional[Tensor]):
                Global conditioning tensor (B, global_channels, 1).
        Returns:
            Tensor:
                Output tensor (B, channels, T).
        """
        if g is not None:
            x = x + g
        for f in self.convs:
            y = f(x * x_mask)
            x = x + y
        return x * x_mask


class ConvFlow(nn.Layer):
    """Convolutional flow module."""

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            kernel_size: int,
            layers: int,
            bins: int=10,
            tail_bound: float=5.0, ):
        """Initialize ConvFlow module.
        Args:
            in_channels (int):
                Number of input channels.
            hidden_channels (int):
                Number of hidden channels.
            kernel_size (int):
                Kernel size.
            layers (int):
                Number of layers.
            bins (int):
                Number of bins.
            tail_bound (float):
                Tail bound value.
        """
        super().__init__()
        self.half_channels = in_channels // 2
        self.hidden_channels = hidden_channels
        self.bins = bins
        self.tail_bound = tail_bound

        self.input_conv = nn.Conv1D(
            self.half_channels,
            hidden_channels,
            1, )
        self.dds_conv = DilatedDepthSeparableConv(
            hidden_channels,
            kernel_size,
            layers,
            dropout_rate=0.0, )
        self.proj = nn.Conv1D(
            hidden_channels,
            self.half_channels * (bins * 3 - 1),
            1, )

        weight = paddle.zeros(paddle.shape(self.proj.weight))

        self.proj.weight = paddle.create_parameter(
            shape=weight.shape,
            dtype=str(weight.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(weight))

        bias = paddle.zeros(paddle.shape(self.proj.bias))

        self.proj.bias = paddle.create_parameter(
            shape=bias.shape,
            dtype=str(bias.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(bias))

    def forward(
            self,
            x: paddle.Tensor,
            x_mask: paddle.Tensor,
            g: Optional[paddle.Tensor]=None,
            inverse: bool=False,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        """Calculate forward propagation.
        Args:
            x (Tensor):
                Input tensor (B, channels, T).
            x_mask (Tensor):
                Mask tensor (B, 1, T).
            g (Optional[Tensor]):
                Global conditioning tensor (B, channels, 1).
            inverse (bool):
                Whether to inverse the flow.
        Returns:
            Tensor:
                Output tensor (B, channels, T).
            Tensor:
                Log-determinant tensor for NLL (B,) if not inverse.
        """
        xa, xb = x.split(2, 1)
        h = self.input_conv(xa)
        h = self.dds_conv(h, x_mask, g=g)
        # (B, half_channels * (bins * 3 - 1), T)
        h = self.proj(h) * x_mask

        b, c, t = xa.shape
        # (B, half_channels, bins * 3 - 1, T) -> (B, half_channels, T, bins * 3 - 1)
        h = h.reshape([b, c, -1, t]).transpose([0, 1, 3, 2])

        denom = math.sqrt(self.hidden_channels)
        unnorm_widths = h[..., :self.bins] / denom
        unnorm_heights = h[..., self.bins:2 * self.bins] / denom
        unnorm_derivatives = h[..., 2 * self.bins:]

        xb, logdet_abs = piecewise_rational_quadratic_transform(
            inputs=xb,
            unnormalized_widths=unnorm_widths,
            unnormalized_heights=unnorm_heights,
            unnormalized_derivatives=unnorm_derivatives,
            inverse=inverse,
            tails="linear",
            tail_bound=self.tail_bound, )
        x = paddle.concat([xa, xb], 1) * x_mask
        logdet = paddle.sum(logdet_abs * x_mask, [1, 2])
        if not inverse:
            return x, logdet
        else:
            return x
