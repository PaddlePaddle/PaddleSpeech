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

"""Multi-Head Attention layer definition."""
import math
import logging
from typing import Optional, Tuple

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

logger = logging.getLogger(__name__)

__all__ = ["MultiHeadedAttention", "RelPositionMultiHeadedAttention"]


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
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
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
        n_batch = query.size(0)
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
            paddle.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.size(0)
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
        x = x.transpose([0, 2, 1, 3]).contiguous().view(
            n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

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
        scores = paddle.matmul(q, k.transpose(
            [0, 1, 3, 2])) / math.sqrt(self.d_k)
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
        self.linear_pos = nn.Linear(n_feat, n_feat, bias_attr=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool=False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros(
            (x.size()[0], x.size()[1], x.size()[2], 1),
            device=x.device,
            dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                pos_emb: torch.Tensor,
                mask: Optional[torch.Tensor]):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)
