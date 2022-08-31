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
"""Decoder self-attention layer definition."""
from typing import Optional
from typing import Tuple

import paddle
from paddle import nn

from paddlespeech.s2t.modules.align import LayerNorm
from paddlespeech.s2t.modules.align import Linear
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ["DecoderLayer"]


class DecoderLayer(nn.Layer):
    """Single decoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (nn.Layer): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (nn.Layer): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (nn.Layer): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(
            self,
            size: int,
            self_attn: nn.Layer,
            src_attn: nn.Layer,
            feed_forward: nn.Layer,
            dropout_rate: float,
            normalize_before: bool=True,
            concat_after: bool=False, ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size, epsilon=1e-12)
        self.norm2 = LayerNorm(size, epsilon=1e-12)
        self.norm3 = LayerNorm(size, epsilon=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear1 = Linear(size + size, size)
        self.concat_linear2 = Linear(size + size, size)

    def forward(
            self,
            tgt: paddle.Tensor,
            tgt_mask: paddle.Tensor,
            memory: paddle.Tensor,
            memory_mask: paddle.Tensor,
            cache: Optional[paddle.Tensor]=None
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute decoded features.
        Args:
            tgt (paddle.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (paddle.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (paddle.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (paddle.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (paddle.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).
        Returns:
            paddle.Tensor: Output tensor (#batch, maxlen_out, size).
            paddle.Tensor: Mask for output tensor (#batch, maxlen_out).
            paddle.Tensor: Encoded memory (#batch, maxlen_in, size).
            paddle.Tensor: Encoded memory mask (#batch, maxlen_in).
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
            ], f"{cache.shape} == {[tgt.shape[0], tgt.shape[1] - 1, self.size]}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            # TODO(Hui Zhang): slice not support bool type
            # tgt_q_mask = tgt_mask[:, -1:, :]
            tgt_q_mask = tgt_mask.cast(paddle.int64)[:, -1:, :].cast(
                paddle.bool)

        if self.concat_after:
            tgt_concat = paddle.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask,
                                       paddle.empty([0]),
                                       paddle.zeros([0, 0, 0, 0]))[0]),
                dim=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(
                self.self_attn(tgt_q, tgt, tgt, tgt_q_mask,
                               paddle.empty([0]), paddle.zeros([0, 0, 0, 0]))[
                                   0])
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = paddle.cat(
                (x, self.src_attn(x, memory, memory, memory_mask,
                                  paddle.empty([0]),
                                  paddle.zeros([0, 0, 0, 0]))[0]),
                dim=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(
                self.src_attn(x, memory, memory, memory_mask,
                              paddle.empty([0]), paddle.zeros([0, 0, 0, 0]))[0])
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = paddle.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask
