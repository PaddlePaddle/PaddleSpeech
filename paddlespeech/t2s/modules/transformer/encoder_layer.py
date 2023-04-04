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
"""Encoder self-attention layer definition."""
import paddle
from paddle import nn


class EncoderLayer(nn.Layer):
    """Encoder layer module.

    Args:
        size (int): 
            Input dimension.
        self_attn (nn.Layer): 
            Self-attention module instance.
            `MultiHeadedAttention`  instance can be used as the argument.
        feed_forward (nn.Layer): 
            Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance can be used as the argument.
        dropout_rate (float): 
            Dropout rate.
        normalize_before (bool): 
            Whether to use layer_norm before the first block.
        concat_after (bool): 
            Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
    """
    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size, bias_attr=True)

    def forward(self, x, mask, cache=None):
        """Compute encoded features.

        Args:
            x(Tensor): 
                Input tensor (#batch, time, size).
            mask(Tensor): 
                Mask tensor for the input (#batch, time).
            cache(Tensor, optional): 
                Cache tensor of the input (#batch, time - 1, size). 

        Returns:
            Tensor: 
                Output tensor (#batch, time, size).
            Tensor: 
                Mask tensor (#batch, time).
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = paddle.concat((x, self.self_attn(x_q, x, x, mask)),
                                     axis=-1)
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

        if cache is not None:
            x = paddle.concat([cache, x], axis=1)

        return x, mask
