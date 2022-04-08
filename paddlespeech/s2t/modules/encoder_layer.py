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
"""Encoder self-attention layer definition."""
from typing import Optional
from typing import Tuple

import paddle
from paddle import nn

from paddlespeech.s2t.modules.align import LayerNorm
from paddlespeech.s2t.modules.align import Linear
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ["TransformerEncoderLayer", "ConformerEncoderLayer"]


class TransformerEncoderLayer(nn.Layer):
    """Encoder layer module."""

    def __init__(
            self,
            size: int,
            self_attn: nn.Layer,
            feed_forward: nn.Layer,
            dropout_rate: float,
            normalize_before: bool=True,
            concat_after: bool=False, ):
        """Construct an EncoderLayer object.

        Args:
            size (int): Input dimension.
            self_attn (nn.Layer): Self-attention module instance.
                `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
                instance can be used as the argument.
            feed_forward (nn.Layer): Feed-forward module instance.
                `PositionwiseFeedForward`, instance can be used as the argument.
            dropout_rate (float): Dropout rate.
            normalize_before (bool):
                True: use layer_norm before each sub-block.
                False: to use layer_norm after each sub-block.
            concat_after (bool): Whether to concat attention layer's input and
                output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size, epsilon=1e-12)
        self.norm2 = LayerNorm(size, epsilon=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        # concat_linear may be not used in forward fuction,
        # but will be saved in the *.pt
        self.concat_linear = Linear(size + size, size)

    def forward(
            self,
            x: paddle.Tensor,
            mask: paddle.Tensor,
            pos_emb: Optional[paddle.Tensor]=None,
            mask_pad: Optional[paddle.Tensor]=None,
            output_cache: Optional[paddle.Tensor]=None,
            cnn_cache: Optional[paddle.Tensor]=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute encoded features.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, size).
            mask (paddle.Tensor): Mask tensor for the input (#batch, time).
            pos_emb (paddle.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (paddle.Tensor): not used here, it's for interface
                compatibility to ConformerEncoderLayer
            output_cache (paddle.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
            cnn_cache (paddle.Tensor): not used here, it's for interface
                compatibility to ConformerEncoderLayer
        Returns:
            paddle.Tensor: Output tensor (#batch, time, size).
            paddle.Tensor: Mask tensor (#batch, time).
            paddle.Tensor: Fake cnn cache tensor for api compatibility with Conformer (#batch, channels, time').
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if output_cache is None:
            x_q = x
        else:
            assert output_cache.shape[0] == x.shape[0]
            assert output_cache.shape[1] < x.shape[1]
            assert output_cache.shape[2] == self.size
            chunk = x.shape[1] - output_cache.shape[1]
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        if self.concat_after:
            x_concat = paddle.concat(
                (x, self.self_attn(x_q, x, x, mask)), axis=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if output_cache is not None:
            x = paddle.concat([output_cache, x], axis=1)

        fake_cnn_cache = paddle.zeros([1], dtype=x.dtype)
        return x, mask, fake_cnn_cache


class ConformerEncoderLayer(nn.Layer):
    """Encoder layer module."""

    def __init__(
            self,
            size: int,
            self_attn: nn.Layer,
            feed_forward: Optional[nn.Layer]=None,
            feed_forward_macaron: Optional[nn.Layer]=None,
            conv_module: Optional[nn.Layer]=None,
            dropout_rate: float=0.1,
            normalize_before: bool=True,
            concat_after: bool=False, ):
        """Construct an EncoderLayer object.

        Args:
            size (int): Input dimension.
            self_attn (nn.Layer): Self-attention module instance.
                `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
                instance can be used as the argument.
            feed_forward (nn.Layer): Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            feed_forward_macaron (nn.Layer): Additional feed-forward module
                instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            conv_module (nn.Layer): Convolution module instance.
                `ConvlutionModule` instance can be used as the argument.
            dropout_rate (float): Dropout rate.
            normalize_before (bool):
                True: use layer_norm before each sub-block.
                False: use layer_norm after each sub-block.
            concat_after (bool): Whether to concat attention layer's input and
                output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size, epsilon=1e-12)  # for the FNN module
        self.norm_mha = LayerNorm(size, epsilon=1e-12)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size, epsilon=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(
                size, epsilon=1e-12)  # for the CNN module
            self.norm_final = LayerNorm(
                size, epsilon=1e-12)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = Linear(size + size, size)

    def forward(
            self,
            x: paddle.Tensor,
            mask: paddle.Tensor,
            pos_emb: paddle.Tensor,
            mask_pad: Optional[paddle.Tensor]=None,
            output_cache: Optional[paddle.Tensor]=None,
            cnn_cache: Optional[paddle.Tensor]=None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute encoded features.
        Args:
            x (paddle.Tensor): (#batch, time, size)
            mask (paddle.Tensor): Mask tensor for the input (#batch, time，time).
            pos_emb (paddle.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (paddle.Tensor): batch padding mask used for conv module, (B, 1, T).
            output_cache (paddle.Tensor): Cache tensor of the encoder output
                (#batch, time2, size), time2 < time in x.
            cnn_cache (paddle.Tensor): Convolution cache in conformer layer
        Returns:
            paddle.Tensor: Output tensor (#batch, time, size).
            paddle.Tensor: Mask tensor (#batch, time).
            paddle.Tensor: New cnn cache tensor (#batch, channels, time').
        """
        # whether to use macaron style FFN
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if output_cache is None:
            x_q = x
        else:
            assert output_cache.shape[0] == x.shape[0]
            assert output_cache.shape[1] < x.shape[1]
            assert output_cache.shape[2] == self.size
            chunk = x.shape[1] - output_cache.shape[1]
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        x_att = self.self_attn(x_q, x, x, pos_emb, mask)

        if self.concat_after:
            x_concat = paddle.concat((x, x_att), axis=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)

        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = paddle.zeros([1], dtype=x.dtype)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)

            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if output_cache is not None:
            x = paddle.concat([output_cache, x], axis=1)

        return x, mask, new_cnn_cache
