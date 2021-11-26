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
"""Decoder self-attention layer definition."""
import paddle
from paddle import nn

from paddlespeech.t2s.modules.layer_norm import LayerNorm


class DecoderLayer(nn.Layer):
    """Single decoder layer module.

    Parameters
    ----------
    size : int
        Input dimension.
    self_attn : nn.Layer
        Self-attention module instance.
        `MultiHeadedAttention` instance can be used as the argument.
    src_attn : nn.Layer
        Self-attention module instance.
        `MultiHeadedAttention` instance can be used as the argument.
    feed_forward : nn.Layer
        Feed-forward module instance.
        `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance can be used as the argument.
    dropout_rate : float
        Dropout rate.
    normalize_before : bool
        Whether to use layer_norm before the first block.
    concat_after : bool
        Whether to concat attention layer's input and output.
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
            self,
            size,
            self_attn,
            src_attn,
            feed_forward,
            dropout_rate,
            normalize_before=True,
            concat_after=False, ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Compute decoded features.

        Parameters
        ----------
        tgt : paddle.Tensor
            Input tensor (#batch, maxlen_out, size).
        tgt_mask : paddle.Tensor
            Mask for input tensor (#batch, maxlen_out).
        memory : paddle.Tensor
            Encoded memory, float32 (#batch, maxlen_in, size).
        memory_mask : paddle.Tensor
            Encoded memory mask (#batch, maxlen_in).
        cache : List[paddle.Tensor]
            List of cached tensors.
            Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns
        ----------
        paddle.Tensor
            Output tensor(#batch, maxlen_out, size).
        paddle.Tensor
            Mask for output tensor (#batch, maxlen_out).
        paddle.Tensor
            Encoded memory (#batch, maxlen_in, size).
        paddle.Tensor
            Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == [
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ], f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_mask = paddle.cast(tgt_mask, dtype="int64")
                tgt_q_mask = tgt_mask[:, -1:, :]
                tgt_q_mask = paddle.cast(tgt_q_mask, dtype="bool")

        if self.concat_after:
            tgt_concat = paddle.concat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), axis=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(
                self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = paddle.concat(
                (x, self.src_attn(x, memory, memory, memory_mask)), axis=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(
                self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = paddle.concat([cache, x], axis=1)

        return x, tgt_mask, memory, memory_mask
