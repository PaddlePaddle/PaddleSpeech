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

from typing import Union
import logging
import numpy as np
import math
from collections import OrderedDict

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

logger = logging.getLogger(__name__)

__all__ = ["brelu", "LinearGLUBlock", "ConstantPad2d", "ConvGLUBlock"]


def brelu(x, t_min=0.0, t_max=24.0, name=None):
    # paddle.to_tensor is dygraph_only can not work under JIT
    t_min = paddle.full(shape=[1], fill_value=t_min, dtype='float32')
    t_max = paddle.full(shape=[1], fill_value=t_max, dtype='float32')
    return x.maximum(t_min).minimum(t_max)


class LinearGLUBlock(nn.Layer):
    """A linear Gated Linear Units (GLU) block."""

    def __init__(self, idim: int):
        """ GLU.
        Args:
            idim (int): input and output dimension
        """
        super().__init__()
        self.fc = nn.Linear(idim, idim * 2)

    def forward(self, xs):
        return glu(self.fc(xs), dim=-1)


# TODO(Hui Zhang): remove this Layer
class ConstantPad2d(nn.Layer):
    """Pads the input tensor boundaries with a constant value.
    For N-dimensional padding, use paddle.nn.functional.pad().
    """

    def __init__(self, padding: Union[tuple, list, int], value: float):
        """
        Args:
            paddle ([tuple]): the size of the padding. 
                If is int, uses the same padding in all boundaries. 
                If a 4-tuple, uses (padding_left, padding_right, padding_top, padding_bottom)
            value ([flaot]): pad value
        """
        self.padding = padding if isinstance(padding,
                                             [tuple, list]) else [padding] * 4
        self.value = value

    def forward(self, xs: paddle.Tensor) -> paddle.Tensor:
        return nn.functional.pad(
            xs,
            self.padding,
            mode='constant',
            value=self.value,
            data_format='NCHW')


class ConvGLUBlock(nn.Layer):
    def __init__(self, kernel_size, in_ch, out_ch, bottlececk_dim=0,
                 dropout=0.):
        """A convolutional Gated Linear Units (GLU) block.

        Args:
            kernel_size (int): kernel size
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            bottlececk_dim (int): dimension of the bottleneck layers for computational efficiency. Defaults to 0.
            dropout (float): dropout probability. Defaults to 0..
        """

        super().__init__()

        self.conv_residual = None
        if in_ch != out_ch:
            self.conv_residual = nn.utils.weight_norm(
                nn.Conv2D(
                    in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1)),
                name='weight',
                dim=0)
            self.dropout_residual = nn.Dropout(p=dropout)

        self.pad_left = ConstantPad2d((0, 0, kernel_size - 1, 0), 0)

        layers = OrderedDict()
        if bottlececk_dim == 0:
            layers['conv'] = nn.utils.weight_norm(
                nn.Conv2D(
                    in_channels=in_ch,
                    out_channels=out_ch * 2,
                    kernel_size=(kernel_size, 1)),
                name='weight',
                dim=0)
            # TODO(hirofumi0810): padding?
            layers['dropout'] = nn.Dropout(p=dropout)
            layers['glu'] = GLU()

        elif bottlececk_dim > 0:
            layers['conv_in'] = nn.utils.weight_norm(
                nn.Conv2D(
                    in_channels=in_ch,
                    out_channels=bottlececk_dim,
                    kernel_size=(1, 1)),
                name='weight',
                dim=0)
            layers['dropout_in'] = nn.Dropout(p=dropout)
            layers['conv_bottleneck'] = nn.utils.weight_norm(
                nn.Conv2D(
                    in_channels=bottlececk_dim,
                    out_channels=bottlececk_dim,
                    kernel_size=(kernel_size, 1)),
                name='weight',
                dim=0)
            layers['dropout'] = nn.Dropout(p=dropout)
            layers['glu'] = GLU()
            layers['conv_out'] = nn.utils.weight_norm(
                nn.Conv2D(
                    in_channels=bottlececk_dim,
                    out_channels=out_ch * 2,
                    kernel_size=(1, 1)),
                name='weight',
                dim=0)
            layers['dropout_out'] = nn.Dropout(p=dropout)

        self.layers = nn.Sequential(layers)

    def forward(self, xs):
        """Forward pass.
        Args:
            xs (FloatTensor): `[B, in_ch, T, feat_dim]`
        Returns:
            out (FloatTensor): `[B, out_ch, T, feat_dim]`
        """
        residual = xs
        if self.conv_residual is not None:
            residual = self.dropout_residual(self.conv_residual(residual))
        xs = self.pad_left(xs)  # `[B, embed_dim, T+kernel-1, 1]`
        xs = self.layers(xs)  # `[B, out_ch * 2, T ,1]`
        xs = xs + residual
        return xs
