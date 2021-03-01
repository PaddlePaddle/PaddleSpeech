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

import logging

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

from deepspeech.modules.mask import sequence_mask
from deepspeech.modules.activation import brelu

logger = logging.getLogger(__name__)

__all__ = ['ConvStack']


class ConvBn(nn.Layer):
    """Convolution layer with batch normalization.

    :param kernel_size: The x dimension of a filter kernel. Or input a tuple for
                        two image dimension.
    :type kernel_size: int|tuple|list
    :param num_channels_in: Number of input channels.
    :type num_channels_in: int
    :param num_channels_out: Number of output channels.
    :type num_channels_out: int
    :param stride: The x dimension of the stride. Or input a tuple for two 
                image dimension. 
    :type stride: int|tuple|list
    :param padding: The x dimension of the padding. Or input a tuple for two
                    image dimension.
    :type padding: int|tuple|list
    :param act: Activation type, relu|brelu
    :type act: string
    :return: Batch norm layer after convolution layer.
    :rtype: Variable

    """

    def __init__(self, num_channels_in, num_channels_out, kernel_size, stride,
                 padding, act):

        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert len(padding) == 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2D(
            num_channels_in,
            num_channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_attr=None,
            bias_attr=False,
            data_format='NCHW')

        self.bn = nn.BatchNorm2D(
            num_channels_out,
            weight_attr=None,
            bias_attr=None,
            data_format='NCHW')
        self.act = F.relu if act == 'relu' else brelu

    def forward(self, x, x_len):
        """
        x(Tensor): audio, shape [B, C, D, T]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        x_len = (x_len - self.kernel_size[1] + 2 * self.padding[1]
                 ) // self.stride[1] + 1

        # reset padding part to 0
        masks = sequence_mask(x_len)  #[B, T]
        masks = masks.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
        x = x.multiply(masks)

        return x, x_len


class ConvStack(nn.Layer):
    """Convolution group with stacked convolution layers.

    :param feat_size: audio feature dim.
    :type feat_size: int
    :param num_stacks: Number of stacked convolution layers.
    :type num_stacks: int
    """

    def __init__(self, feat_size, num_stacks):
        super().__init__()
        self.feat_size = feat_size  # D
        self.num_stacks = num_stacks

        self.conv_in = ConvBn(
            num_channels_in=1,
            num_channels_out=32,
            kernel_size=(41, 11),  #[D, T]
            stride=(2, 3),
            padding=(20, 5),
            act='brelu')

        out_channel = 32
        self.conv_stack = nn.LayerList([
            ConvBn(
                num_channels_in=32,
                num_channels_out=out_channel,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
                act='brelu') for i in range(num_stacks - 1)
        ])

        # conv output feat_dim
        output_height = (feat_size - 1) // 2 + 1
        for i in range(self.num_stacks - 1):
            output_height = (output_height - 1) // 2 + 1
        self.output_height = out_channel * output_height

    def forward(self, x, x_len):
        """
        x: shape [B, C, D, T]
        x_len : shape [B]
        """
        x, x_len = self.conv_in(x, x_len)
        for i, conv in enumerate(self.conv_stack):
            x, x_len = conv(x, x_len)
        return x, x_len
