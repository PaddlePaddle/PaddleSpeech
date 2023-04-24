# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Mobvoi Inc. All Rights Reserved.
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
# Modified from wenet(https://github.com/wenet-e2e/wenet)
"""Subsampling layer definition."""
from typing import Tuple

import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.s2t import masked_fill
from paddlespeech.s2t.modules.align import Conv1D
from paddlespeech.s2t.modules.conv2d import Conv2DValid
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = [
    "TimeReductionLayerStream", "TimeReductionLayer1D", "TimeReductionLayer2D"
]


class TimeReductionLayer1D(nn.Layer):
    """
    Modified NeMo,
    Squeezeformer Time Reduction procedure.
    Downsamples the audio by `stride` in the time dimension.
    Args:
        channel (int): input dimension of
                       MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for
                           depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    def __init__(self,
                 channel: int,
                 out_dim: int,
                 kernel_size: int=5,
                 stride: int=2):
        super(TimeReductionLayer1D, self).__init__()

        self.channel = channel
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = max(0, self.kernel_size - self.stride)

        self.dw_conv = Conv1D(
            in_channels=channel,
            out_channels=channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            groups=channel, )

        self.pw_conv = Conv1D(
            in_channels=channel,
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1, )

        self.init_weights()

    def init_weights(self):
        dw_max = self.kernel_size**-0.5
        pw_max = self.channel**-0.5
        self.dw_conv._param_attr = paddle.nn.initializer.Uniform(
            low=-dw_max, high=dw_max)
        self.dw_conv._bias_attr = paddle.nn.initializer.Uniform(
            low=-dw_max, high=dw_max)
        self.pw_conv._param_attr = paddle.nn.initializer.Uniform(
            low=-pw_max, high=pw_max)
        self.pw_conv._bias_attr = paddle.nn.initializer.Uniform(
            low=-pw_max, high=pw_max)

    def forward(
            self,
            xs,
            xs_lens: paddle.Tensor,
            mask: paddle.Tensor=paddle.ones((0, 0, 0), dtype=paddle.bool),
            mask_pad: paddle.Tensor=paddle.ones((0, 0, 0),
                                                dtype=paddle.bool), ):
        xs = xs.transpose([0, 2, 1])  # [B, C, T]
        xs = masked_fill(xs, mask_pad.equal(0), 0.0)

        xs = self.dw_conv(xs)
        xs = self.pw_conv(xs)

        xs = xs.transpose([0, 2, 1])  # [B, T, C]

        B, T, D = xs.shape
        mask = mask[:, ::self.stride, ::self.stride]
        mask_pad = mask_pad[:, :, ::self.stride]
        L = mask_pad.shape[-1]
        # For JIT exporting, we remove F.pad operator.
        if L - T < 0:
            xs = xs[:, :L - T, :]
        else:
            dummy_pad = paddle.zeros([B, L - T, D], dtype=paddle.float32)
            xs = paddle.concat([xs, dummy_pad], axis=1)

        xs_lens = (xs_lens + 1) // 2
        return xs, xs_lens, mask, mask_pad


class TimeReductionLayer2D(nn.Layer):
    def __init__(self, kernel_size: int=5, stride: int=2, encoder_dim: int=256):
        super(TimeReductionLayer2D, self).__init__()
        self.encoder_dim = encoder_dim
        self.kernel_size = kernel_size
        self.dw_conv = Conv2DValid(
            in_channels=encoder_dim,
            out_channels=encoder_dim,
            kernel_size=(kernel_size, 1),
            stride=stride,
            valid_trigy=True)
        self.pw_conv = Conv2DValid(
            in_channels=encoder_dim,
            out_channels=encoder_dim,
            kernel_size=1,
            stride=1,
            valid_trigx=False,
            valid_trigy=False)

        self.kernel_size = kernel_size
        self.stride = stride
        self.init_weights()

    def init_weights(self):
        dw_max = self.kernel_size**-0.5
        pw_max = self.encoder_dim**-0.5
        self.dw_conv._param_attr = paddle.nn.initializer.Uniform(
            low=-dw_max, high=dw_max)
        self.dw_conv._bias_attr = paddle.nn.initializer.Uniform(
            low=-dw_max, high=dw_max)
        self.pw_conv._param_attr = paddle.nn.initializer.Uniform(
            low=-pw_max, high=pw_max)
        self.pw_conv._bias_attr = paddle.nn.initializer.Uniform(
            low=-pw_max, high=pw_max)

    def forward(
            self,
            xs: paddle.Tensor,
            xs_lens: paddle.Tensor,
            mask: paddle.Tensor=paddle.ones((0, 0, 0), dtype=paddle.bool),
            mask_pad: paddle.Tensor=paddle.ones((0, 0, 0), dtype=paddle.bool),
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        xs = masked_fill(xs, mask_pad.transpose([0, 2, 1]).equal(0), 0.0)
        xs = xs.unsqueeze(1)
        padding1 = self.kernel_size - self.stride
        xs = F.pad(
            xs, (0, 0, 0, 0, 0, padding1, 0, 0), mode='constant', value=0.)
        xs = self.dw_conv(xs.transpose([0, 3, 2, 1]))
        xs = self.pw_conv(xs).transpose([0, 3, 2, 1]).squeeze(1)
        tmp_length = xs.shape[1]
        xs_lens = (xs_lens + 1) // 2
        padding2 = max(0, (xs_lens.max() - tmp_length).item())
        batch_size, hidden = xs.shape[0], xs.shape[-1]
        dummy_pad = paddle.zeros(
            [batch_size, padding2, hidden], dtype=paddle.float32)
        xs = paddle.concat([xs, dummy_pad], axis=1)
        mask = mask[:, ::2, ::2]
        mask_pad = mask_pad[:, :, ::2]
        return xs, xs_lens, mask, mask_pad


class TimeReductionLayerStream(nn.Layer):
    """
    Squeezeformer Time Reduction procedure.
    Downsamples the audio by `stride` in the time dimension.
    Args:
        channel (int): input dimension of
            MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for
            depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    def __init__(self,
                 channel: int,
                 out_dim: int,
                 kernel_size: int=1,
                 stride: int=2):
        super(TimeReductionLayerStream, self).__init__()

        self.channel = channel
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride

        self.dw_conv = Conv1D(
            in_channels=channel,
            out_channels=channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=channel)

        self.pw_conv = Conv1D(
            in_channels=channel,
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
        self.init_weights()

    def init_weights(self):
        dw_max = self.kernel_size**-0.5
        pw_max = self.channel**-0.5
        self.dw_conv._param_attr = paddle.nn.initializer.Uniform(
            low=-dw_max, high=dw_max)
        self.dw_conv._bias_attr = paddle.nn.initializer.Uniform(
            low=-dw_max, high=dw_max)
        self.pw_conv._param_attr = paddle.nn.initializer.Uniform(
            low=-pw_max, high=pw_max)
        self.pw_conv._bias_attr = paddle.nn.initializer.Uniform(
            low=-pw_max, high=pw_max)

    def forward(
            self,
            xs,
            xs_lens: paddle.Tensor,
            mask: paddle.Tensor=paddle.ones([0, 0, 0], dtype=paddle.bool),
            mask_pad: paddle.Tensor=paddle.ones([0, 0, 0], dtype=paddle.bool)):
        xs = xs.transpose([0, 2, 1])  # [B, C, T]
        xs = masked_fill(xs, mask_pad.equal(0), 0.0)

        xs = self.dw_conv(xs)
        xs = self.pw_conv(xs)

        xs = xs.transpose([0, 2, 1])  # [B, T, C]

        B, T, D = xs.shape
        mask = mask[:, ::self.stride, ::self.stride]
        mask_pad = mask_pad[:, :, ::self.stride]
        L = mask_pad.shape[-1]
        # For JIT exporting, we remove F.pad operator.
        if L - T < 0:
            xs = xs[:, :L - T, :]
        else:
            dummy_pad = paddle.zeros([B, L - T, D], dtype=paddle.float32)
            xs = paddle.concat([xs, dummy_pad], axis=1)

        xs_lens = (xs_lens + 1) // 2
        return xs, xs_lens, mask, mask_pad
