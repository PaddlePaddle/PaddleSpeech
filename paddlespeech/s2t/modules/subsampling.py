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
from paddlespeech.s2t.modules.align import Conv2D
from paddlespeech.s2t.modules.align import LayerNorm
from paddlespeech.s2t.modules.align import Linear
from paddlespeech.s2t.modules.conv2d import Conv2DValid
from paddlespeech.s2t.modules.embedding import PositionalEncoding
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = [
    "LinearNoSubsampling", "Conv2dSubsampling4", "Conv2dSubsampling6",
    "Conv2dSubsampling8", "TimeReductionLayerStream", "TimeReductionLayer1D",
    "TimeReductionLayer2D", "DepthwiseConv2DSubsampling4"
]


class BaseSubsampling(nn.Layer):
    def __init__(self, pos_enc_class: nn.Layer=PositionalEncoding):
        super().__init__()
        self.pos_enc = pos_enc_class
        # window size = (1 + right_context) + (chunk_size -1) * subsampling_rate
        self.right_context = 0
        # stride = subsampling_rate * chunk_size
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> paddle.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Layer=PositionalEncoding):
        """Construct an linear object.
        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
            pos_enc_class (PositionalEncoding): position encoding class
        """
        super().__init__(pos_enc_class)
        self.out = nn.Sequential(
            Linear(idim, odim),
            LayerNorm(odim, epsilon=1e-12),
            nn.Dropout(dropout_rate),
            nn.ReLU(), )
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset: int=0
                ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Input x.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, idim).
            x_mask (paddle.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            paddle.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            paddle.Tensor: positional encoding
            paddle.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .
        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv2dSubsampling(BaseSubsampling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Conv2dSubsampling4(Conv2dSubsampling):
    """Convolutional 2D subsampling (to 1/4 length)."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Layer=PositionalEncoding):
        """Construct an Conv2dSubsampling4 object.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
        """
        super().__init__(pos_enc_class)
        self.conv = nn.Sequential(
            Conv2D(1, odim, 3, 2),
            nn.ReLU(),
            Conv2D(odim, odim, 3, 2),
            nn.ReLU(), )
        self.out = nn.Sequential(
            Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.subsampling_rate = 4
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset: int=0
                ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Subsample x.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, idim).
            x_mask (paddle.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            paddle.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            paddle.Tensor: positional encoding
            paddle.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.out(x.transpose([0, 2, 1, 3]).reshape([b, -1, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling6(Conv2dSubsampling):
    """Convolutional 2D subsampling (to 1/6 length)."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Layer=PositionalEncoding):
        """Construct an Conv2dSubsampling6 object.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
            pos_enc (PositionalEncoding): Custom position encoding layer.
        """
        super().__init__(pos_enc_class)
        self.conv = nn.Sequential(
            Conv2D(1, odim, 3, 2),
            nn.ReLU(),
            Conv2D(odim, odim, 5, 3),
            nn.ReLU(), )
        # O = (I - F + Pstart + Pend) // S + 1
        # when Padding == 0, O = (I - F - S) // S
        self.linear = Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset: int=0
                ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Subsample x.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, idim).
            x_mask (paddle.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            paddle.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            paddle.Tensor: positional encoding
            paddle.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.linear(x.transpose([0, 2, 1, 3]).reshape([b, -1, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(Conv2dSubsampling):
    """Convolutional 2D subsampling (to 1/8 length)."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Layer=PositionalEncoding):
        """Construct an Conv2dSubsampling8 object.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
        """
        super().__init__(pos_enc_class)
        self.conv = nn.Sequential(
            Conv2D(1, odim, 3, 2),
            nn.ReLU(),
            Conv2D(odim, odim, 3, 2),
            nn.ReLU(),
            Conv2D(odim, odim, 3, 2),
            nn.ReLU(), )
        self.linear = Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2),
                             odim)
        self.subsampling_rate = 8
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset: int=0
                ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Subsample x.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, idim).
            x_mask (paddle.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            paddle.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            paddle.Tensor: positional encoding
            paddle.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.linear(x.transpose([0, 2, 1, 3]).reshape([b, -1, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]


class DepthwiseConv2DSubsampling4(BaseSubsampling):
    """Depthwise Convolutional 2D subsampling (to 1/4 length).

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            pos_enc_class (nn.Layer): position encoding class.
            dw_stride (int): Whether do depthwise convolution.
            input_size (int): filter bank dimension.

        """

    def __init__(self,
                 idim: int,
                 odim: int,
                 pos_enc_class: nn.Layer,
                 dw_stride: bool=False,
                 input_size: int=80,
                 input_dropout_rate: float=0.1,
                 init_weights: bool=True):
        super(DepthwiseConv2DSubsampling4, self).__init__()
        self.idim = idim
        self.odim = odim
        self.pw_conv = Conv2D(
            in_channels=idim, out_channels=odim, kernel_size=3, stride=2)
        self.act1 = nn.ReLU()
        self.dw_conv = Conv2D(
            in_channels=odim,
            out_channels=odim,
            kernel_size=3,
            stride=2,
            groups=odim if dw_stride else 1)
        self.act2 = nn.ReLU()
        self.pos_enc = pos_enc_class
        self.input_proj = nn.Sequential(
            Linear(odim * (((input_size - 1) // 2 - 1) // 2), odim),
            nn.Dropout(p=input_dropout_rate))
        if init_weights:
            linear_max = (odim * input_size / 4)**-0.5
            self.input_proj.state_dict()[
                '0.weight'] = paddle.nn.initializer.Uniform(
                    low=-linear_max, high=linear_max)
            self.input_proj.state_dict()[
                '0.bias'] = paddle.nn.initializer.Uniform(
                    low=-linear_max, high=linear_max)

        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset: int=0
                ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.pw_conv(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.act2(x)
        b, c, t, f = x.shape
        x = x.transpose([0, 2, 1, 3]).reshape([b, -1, c * f])
        x, pos_emb = self.pos_enc(x, offset)
        x = self.input_proj(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


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
