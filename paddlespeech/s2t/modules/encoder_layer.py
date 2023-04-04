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

__all__ = [
    "TransformerEncoderLayer", "ConformerEncoderLayer",
    "SqueezeformerEncoderLayer"
]


class TransformerEncoderLayer(nn.Layer):
    """Encoder layer module."""
    def __init__(
        self,
        size: int,
        self_attn: nn.Layer,
        feed_forward: nn.Layer,
        dropout_rate: float,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
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
        pos_emb: paddle.Tensor,
        mask_pad: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool),
        att_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0]),
        cnn_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0])
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute encoded features.
        Args:
            x (paddle.Tensor): (#batch, time, size)
            mask (paddle.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (paddle.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (paddle.Tensor): does not used in transformer layer,
                just for unified api with conformer.
            att_cache (paddle.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (paddle.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2), not used here, it's for interface
                compatibility to ConformerEncoderLayer.
        Returns:
            paddle.Tensor: Output tensor (#batch, time, size).
            paddle.Tensor: Mask tensor (#batch, time, time).
            paddle.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            paddle.Tensor: cnn_cahce tensor (#batch=1, size, cache_t2).
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        x_att, new_att_cache = self.self_attn(x, x, x, mask, cache=att_cache)

        if self.concat_after:
            x_concat = paddle.concat((x, x_att), axis=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        fake_cnn_cache = paddle.zeros([0, 0, 0], dtype=x.dtype)
        return x, mask, new_att_cache, fake_cnn_cache


class ConformerEncoderLayer(nn.Layer):
    """Encoder layer module."""
    def __init__(
        self,
        size: int,
        self_attn: nn.Layer,
        feed_forward: Optional[nn.Layer] = None,
        feed_forward_macaron: Optional[nn.Layer] = None,
        conv_module: Optional[nn.Layer] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
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
            self.norm_conv = LayerNorm(size,
                                       epsilon=1e-12)  # for the CNN module
            self.norm_final = LayerNorm(
                size, epsilon=1e-12)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    def forward(
        self,
        x: paddle.Tensor,
        mask: paddle.Tensor,
        pos_emb: paddle.Tensor,
        mask_pad: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool),
        att_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0]),
        cnn_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0])
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute encoded features.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, size).
            mask (paddle.Tensor): Mask tensor for the input (#batch, time, time).
                (0,0,0) means fake mask.
            pos_emb (paddle.Tensor): postional encoding, must not be None 
                for ConformerEncoderLayer
            mask_pad (paddle.Tensor): batch padding mask used for conv module.
               (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (paddle.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (paddle.Tensor): Convolution cache in conformer layer
                (1, #batch=1, size, cache_t2). First dim will not be used, just
                for dy2st.
        Returns:
           paddle.Tensor: Output tensor (#batch, time, size).
           paddle.Tensor: Mask tensor (#batch, time, time).
           paddle.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
           paddle.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """
        # (1, #batch=1, size, cache_t2) -> (#batch=1, size, cache_t2)
        cnn_cache = paddle.squeeze(cnn_cache, axis=0)

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

        x_att, new_att_cache = self.self_attn(x,
                                              x,
                                              x,
                                              mask,
                                              pos_emb,
                                              cache=att_cache)

        if self.concat_after:
            x_concat = paddle.concat((x, x_att), axis=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)

        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = paddle.zeros([0, 0, 0], dtype=x.dtype)
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

        return x, mask, new_att_cache, new_cnn_cache


class SqueezeformerEncoderLayer(nn.Layer):
    """Encoder layer module."""
    def __init__(self,
                 size: int,
                 self_attn: paddle.nn.Layer,
                 feed_forward1: Optional[nn.Layer] = None,
                 conv_module: Optional[nn.Layer] = None,
                 feed_forward2: Optional[nn.Layer] = None,
                 normalize_before: bool = False,
                 dropout_rate: float = 0.1,
                 concat_after: bool = False):
        """Construct an EncoderLayer object.

        Args:
            size (int): Input dimension.
            self_attn (paddle.nn.Layer): Self-attention module instance.
                `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
                instance can be used as the argument.
            feed_forward1 (paddle.nn.Layer): Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            conv_module (paddle.nn.Layer): Convolution module instance.
                `ConvlutionLayer` instance can be used as the argument.
            feed_forward2 (paddle.nn.Layer): Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            dropout_rate (float): Dropout rate.
            normalize_before (bool):
                True: use layer_norm before each sub-block.
                False: use layer_norm after each sub-block.
        """
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.layer_norm1 = LayerNorm(size)
        self.ffn1 = feed_forward1
        self.layer_norm2 = LayerNorm(size)
        self.conv_module = conv_module
        self.layer_norm3 = LayerNorm(size)
        self.ffn2 = feed_forward2
        self.layer_norm4 = LayerNorm(size)
        self.normalize_before = normalize_before
        self.dropout = nn.Dropout(dropout_rate)
        self.concat_after = concat_after
        if concat_after:
            self.concat_linear = Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    def forward(
        self,
        x: paddle.Tensor,
        mask: paddle.Tensor,
        pos_emb: paddle.Tensor,
        mask_pad: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool),
        att_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0]),
        cnn_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0]),
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute encoded features.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, size).
            mask (paddle.Tensor): Mask tensor for the input (#batch, time, time).
                (0,0,0) means fake mask.
            pos_emb (paddle.Tensor): postional encoding, must not be None
                for ConformerEncoderLayer
            mask_pad (paddle.Tensor): batch padding mask used for conv module.
               (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (paddle.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (paddle.Tensor): Convolution cache in conformer layer
                (1, #batch=1, size, cache_t2). First dim will not be used, just
                for dy2st.
        Returns:
           paddle.Tensor: Output tensor (#batch, time, size).
           paddle.Tensor: Mask tensor (#batch, time, time).
           paddle.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
           paddle.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """
        # self attention module
        residual = x
        if self.normalize_before:
            x = self.layer_norm1(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        if self.concat_after:
            x_concat = paddle.concat((x, x_att), axis=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.layer_norm1(x)

        # ffn module
        residual = x
        if self.normalize_before:
            x = self.layer_norm2(x)
        x = self.ffn1(x)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm2(x)

        # conv module
        residual = x
        if self.normalize_before:
            x = self.layer_norm3(x)
        x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm3(x)

        # ffn module
        residual = x
        if self.normalize_before:
            x = self.layer_norm4(x)
        x = self.ffn2(x)
        # we do not use dropout here since it is inside feed forward function
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm4(x)

        return x, mask, new_att_cache, new_cnn_cache
