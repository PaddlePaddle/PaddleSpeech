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
"""Multi-Head Attention layer definition."""
import math
from typing import Optional
from typing import Tuple

import paddle
from paddle import nn
from paddle.nn import initializer as I

from paddlespeech.s2t.modules.align import Linear
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ["MultiHeadedAttention", "RelPositionMultiHeadedAttention"]

# Relative Positional Encodings
# https://www.jianshu.com/p/c0608efcc26f
# https://zhuanlan.zhihu.com/p/344604604


class MultiHeadedAttention(nn.Layer):
    """Multi-Head Attention layer."""

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object.
        Args:
            n_head (int): The number of heads.
            n_feat (int): The number of features.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = Linear(n_feat, n_feat)
        self.linear_k = Linear(n_feat, n_feat)
        self.linear_v = Linear(n_feat, n_feat)
        self.linear_out = Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self,
                    query: paddle.Tensor,
                    key: paddle.Tensor,
                    value: paddle.Tensor
                    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Transform query, key and value.
        Args:
            query (paddle.Tensor): Query tensor (#batch, time1, size).
            key (paddle.Tensor): Key tensor (#batch, time2, size).
            value (paddle.Tensor): Value tensor (#batch, time2, size).
        Returns:
            paddle.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            paddle.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            paddle.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).
        """
        n_batch = query.shape[0]
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose([0, 2, 1, 3])  # (batch, head, time1, d_k)
        k = k.transpose([0, 2, 1, 3])  # (batch, head, time2, d_k)
        v = v.transpose([0, 2, 1, 3])  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self,
                          value: paddle.Tensor,
                          scores: paddle.Tensor,
                          mask: Optional[paddle.Tensor]) -> paddle.Tensor:
        """Compute attention context vector.
        Args:
            value (paddle.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (paddle.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (paddle.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            paddle.Tensor: Transformed value weighted
                by the attention score, (#batch, time1, d_model).
        """
        n_batch = value.shape[0]
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = paddle.softmax(
                scores, axis=-1).masked_fill(mask,
                                             0.0)  # (batch, head, time1, time2)
        else:
            attn = paddle.softmax(
                scores, axis=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = paddle.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose([0, 2, 1, 3]).view(n_batch, -1, self.h *
                                           self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self,
                query: paddle.Tensor,
                key: paddle.Tensor,
                value: paddle.Tensor,
                mask: Optional[paddle.Tensor]) -> paddle.Tensor:
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = paddle.matmul(q,
                               k.transpose([0, 1, 3, 2])) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding."""

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object.
        Paper: https://arxiv.org/abs/1901.02860
        Args:
            n_head (int): The number of heads.
            n_feat (int): The number of features.
            dropout_rate (float): Dropout rate.
        """
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = Linear(n_feat, n_feat, bias_attr=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        #self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        #self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        #torch.nn.init.xavier_uniform_(self.pos_bias_u)
        #torch.nn.init.xavier_uniform_(self.pos_bias_v)
        pos_bias_u = self.create_parameter(
            [self.h, self.d_k], default_initializer=I.XavierUniform())
        self.add_parameter('pos_bias_u', pos_bias_u)
        pos_bias_v = self.create_parameter(
            (self.h, self.d_k), default_initializer=I.XavierUniform())
        self.add_parameter('pos_bias_v', pos_bias_v)

    def rel_shift(self, x, zero_triu: bool=False):
        """Compute relative positinal encoding.
        Args:
            x (paddle.Tensor): Input tensor (batch, head, time1, time1).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            paddle.Tensor: Output tensor. (batch, head, time1, time1)
        """
        zero_pad = paddle.zeros(
            (x.shape[0], x.shape[1], x.shape[2], 1), dtype=x.dtype)
        x_padded = paddle.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.shape[0], x.shape[1], x.shape[3] + 1,
                                 x.shape[2])
        x = x_padded[:, :, 1:].view_as(x)  # [B, H, T1, T1]

        if zero_triu:
            ones = paddle.ones((x.shape[2], x.shape[3]))
            x = x * paddle.tril(ones, x.shape[3] - x.shape[2])[None, None, :, :]

        return x

    def forward(self,
                query: paddle.Tensor,
                key: paddle.Tensor,
                value: paddle.Tensor,
                pos_emb: paddle.Tensor,
                mask: Optional[paddle.Tensor]):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (paddle.Tensor): Query tensor (#batch, time1, size).
            key (paddle.Tensor): Key tensor (#batch, time2, size).
            value (paddle.Tensor): Value tensor (#batch, time2, size).
            pos_emb (paddle.Tensor): Positional embedding tensor
                (#batch, time1, size).
            mask (paddle.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            paddle.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose([0, 2, 1, 3])  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose([0, 2, 1, 3])  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose([0, 2, 1, 3])
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose([0, 2, 1, 3])

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = paddle.matmul(q_with_bias_u, k.transpose([0, 1, 3, 2]))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = paddle.matmul(q_with_bias_v, p.transpose([0, 1, 3, 2]))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)
