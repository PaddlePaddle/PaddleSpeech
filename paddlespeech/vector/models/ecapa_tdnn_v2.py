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

"""An implementation of the speaker embedding model in a paper.
"ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).
"""

import sys
import argparse
import math
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.nn import Conv1D, MaxPool2D, Linear
import numpy as np

from paddlespeech.vector.layer.layers import AdditiveMarginLinear, SEBlock
from paddlespeech.vector.utils.data_utils import length_to_mask
from paddlespeech.vector import _logger as log

class AttentiveStatisticsPooling(nn.Layer):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1D(
            attention_channels, channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = paddle.shape(x)[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = paddle.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clip(eps)
            )
            return mean, std

        if lengths is None:
            lengths = paddle.ones([x.shape[0]])

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, dtype="int64")
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            float_mask = paddle.cast(mask, dtype="float32")
            total = float_mask.sum(axis=2, keepdim=True)
            mean, std = _compute_statistics(x, float_mask / total)
            mean = mean.unsqueeze(2).expand([mean.shape[0], mean.shape[1], L])
            std = std.unsqueeze(2).expand([mean.shape[0], mean.shape[1], L])
            attn = paddle.concat([x, mean, std], axis=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        neg_inf = paddle.ones_like(attn) * float("-inf")
        mask = mask.expand(attn.shape)
        attn = paddle.where(mask == 0, neg_inf, attn)

        attn = F.softmax(attn, axis=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = paddle.concat((mean, std), axis=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class TDNNBlock(nn.Layer):
    """
    TDNN conv1d block
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 activation=nn.ReLU):
        super(TDNNBlock, self).__init__()
        self.conv = nn.Conv1D(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding="SAME", padding_mode="zeros",
                              weight_attr=nn.initializer.KaimingNormal())
        self.activation = activation()
        self.bn = nn.BatchNorm1D(out_channels)

    def forward(self, x):
        """
        Forward inference for the block

        Args:
            x: input tensor

        Returns:
            x: output tensor
        """
        return self.bn(self.activation(self.conv(x)))


class Res2NetBlock(nn.Layer):
    """An implementation of Res2NetBlock w/ dilation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale=8,
                 dilation=1):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.LayerList(
            [
                TDNNBlock(
                    in_channel, hidden_channel, kernel_size=3, dilation=dilation
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        """
        Forward propagationg
        """
        chunk_list = paddle.chunk(x, self.scale, axis=1)
        y_i = self.blocks[0](chunk_list[1])
        ys = list([chunk_list[0], y_i])
        for i, x_i in enumerate(chunk_list[2:]):
            y_i = self.blocks[i + 1](x_i + y_i)
            ys.append(y_i)
        y = paddle.concat(ys, axis=1)
        return y


class SERes2NetBlock(nn.Layer):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, dilation
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1D(
                in_channels,
                out_channels,
                kernel_size=1,
            )

    def forward(self, x, lengths=None):
        """
        Forward propagationg
        """
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)

        return x + residual


class ECAPATDNN(nn.Layer):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).
    """
    #512, 512, 512, 512, 1536
    # 1024, 1024, 1024, 1024, 3072
    def __init__(
        self,
        input_size,
        out_neurons,
        lin_neurons=192,
        activation=nn.ReLU,
        channels=[512, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        mode="train"
    ):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.seres2net_blocks = nn.LayerList()
        self.mode = mode

        # The initial TDNN layer
        self.tdnn1 = TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.seres2net_blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
                channels[-1],
                channels[-1],
                kernel_sizes[-1],
                dilations[-1],
                activation,
        )

        # Attentitve Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
                channels[-1],
                attention_channels=attention_channels,
                global_context=global_context,
        )

        self.bn = nn.BatchNorm1D(channels[-1] * 2)

        # Final linear transformation
        self.fc1 = nn.Conv1D(
                channels[-1] * 2,
                lin_neurons,
                kernel_size=1,
        )

        self.fc_blocks = nn.LayerList()
        if mode == "train":
            self.fc_blocks.append(AdditiveMarginLinear(lin_neurons, out_neurons))

    def forward(self, x, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency

        x = self.tdnn1(x)
        xl = [x]
        for layer in self.seres2net_blocks:
            x = layer(x, lengths=lengths)
            xl.append(x)

        # Multi-layer feature aggregation
        x = paddle.concat(xl[1:], axis=1)

        x = self.mfa(x)
        x = self.asp(x, lengths)
        x = self.bn(x)
        x = self.fc1(x)

        x = x.squeeze(axis=2)

        for layer in self.fc_blocks:
            x = layer(x)

        return x

    @staticmethod
    def add_specific_args(parser):
        """
        Static class method for xvector parameters configuration
        """
        # parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_in", action="store", type=int,
                            help="n_in(int): input feature dim")
        parser.add_argument("--n_out", action="store", type=int,
                            help="n_out(int): output feature dim")

        return parser

