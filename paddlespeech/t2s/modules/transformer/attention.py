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
"""Multi-Head Attention layer definition."""
import math

import numpy
import paddle
from paddle import nn

from paddlespeech.t2s.modules.masked_fill import masked_fill


class MultiHeadedAttention(nn.Layer):
    """Multi-Head Attention layer.
    Args:
        n_head (int): 
            The number of heads.
        n_feat (int): 
            The number of features.
        dropout_rate (float): 
            Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias_attr=True)
        self.linear_k = nn.Linear(n_feat, n_feat, bias_attr=True)
        self.linear_v = nn.Linear(n_feat, n_feat, bias_attr=True)
        self.linear_out = nn.Linear(n_feat, n_feat, bias_attr=True)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query(Tensor): 
                query tensor (#batch, time1, size).
            key(Tensor): 
                Key tensor (#batch, time2, size).
            value(Tensor): 
                Value tensor (#batch, time2, size).

        Returns:
            Tensor: 
                Transformed query tensor (#batch, n_head, time1, d_k).
            Tensor: 
                Transformed key tensor (#batch, n_head, time2, d_k).
            Tensor: 
                Transformed value tensor (#batch, n_head, time2, d_k).
        """
        n_batch = paddle.shape(query)[0]

        q = paddle.reshape(
            self.linear_q(query), [n_batch, -1, self.h, self.d_k])
        k = paddle.reshape(self.linear_k(key), [n_batch, -1, self.h, self.d_k])
        v = paddle.reshape(
            self.linear_v(value), [n_batch, -1, self.h, self.d_k])

        # (batch, head, time1, d_k)
        q = q.transpose((0, 2, 1, 3))
        # (batch, head, time2, d_k)
        k = k.transpose((0, 2, 1, 3))
        # (batch, head, time2, d_k)
        v = v.transpose((0, 2, 1, 3))
        return q, k, v

    def forward_attention(self, value, scores, mask=None):
        """Compute attention context vector.

        Args:
            value(Tensor): 
                Transformed value (#batch, n_head, time2, d_k).
            scores(Tensor): 
                Attention score (#batch, n_head, time1, time2).
            mask(Tensor, optional): 
                Mask (#batch, 1, time2) or (#batch, time1, time2). (Default value = None)

        Returns:
            Tensor: Transformed value (#batch, time1, d_model) weighted by the attention score (#batch, time1, time2).
        """
        n_batch = paddle.shape(value)[0]
        softmax = paddle.nn.Softmax(axis=-1)
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = paddle.logical_not(mask)
            # assume scores.dtype==paddle.float32, we only use "float32" here
            dtype = str(scores.dtype).split(".")[-1]
            min_value = numpy.finfo(dtype).min
            scores = masked_fill(scores, mask, min_value)
            # (batch, head, time1, time2)
            self.attn = softmax(scores)
            self.attn = masked_fill(self.attn, mask, 0.0)
        else:
            # (batch, head, time1, time2)
            self.attn = softmax(scores)
            # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn)
        # (batch, head, time1, time2) * (batch, head, time2, d_k) -> # (batch, head, time1, d_k)
        x = paddle.matmul(p_attn, value)
        # (batch, time1, d_model)
        x = (paddle.reshape(
            x.transpose((0, 2, 1, 3)), (n_batch, -1, self.h * self.d_k)))
        # (batch, time1, d_model)
        return self.linear_out(x)

    def forward(self, query, key, value, mask=None):
        """Compute scaled dot product attention.

        Args:
            query(Tensor): 
                Query tensor (#batch, time1, size).
            key(Tensor): 
                Key tensor (#batch, time2, size).
            value(Tensor): 
                Value tensor (#batch, time2, size).
            mask(Tensor, optional): 
                Mask tensor (#batch, 1, time2) or (#batch, time1, time2). (Default value = None)

        Returns:
            Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = paddle.matmul(q, k.transpose(
            (0, 1, 3, 2))) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): 
            The number of heads.
        n_feat (int): 
            The number of features.
        dropout_rate (float): 
            Dropout rate.
        zero_triu (bool): 
            Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias_attr=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3

        self.pos_bias_u = paddle.create_parameter(
            shape=(self.h, self.d_k),
            dtype='float32',
            default_initializer=paddle.nn.initializer.XavierUniform())
        self.pos_bias_v = paddle.create_parameter(
            shape=(self.h, self.d_k),
            dtype='float32',
            default_initializer=paddle.nn.initializer.XavierUniform())

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x(Tensor): 
                Input tensor (batch, head, time1, 2*time1-1).

        Returns:
            Tensor: Output tensor.
        """
        b, h, t1, t2 = paddle.shape(x)
        zero_pad = paddle.zeros((b, h, t1, 1))
        x_padded = paddle.concat([zero_pad, x], axis=-1)
        x_padded = x_padded.reshape([b, h, t2 + 1, t1])
        # only keep the positions from 0 to time2
        x = x_padded[:, :, 1:].reshape([b, h, t1, t2])[:, :, :, :t2 // 2 + 1]

        if self.zero_triu:
            ones = paddle.ones((t1, t2))
            x = x * paddle.tril(ones, t2 - t1)[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query(Tensor): 
                Query tensor (#batch, time1, size).
            key(Tensor): 
                Key tensor (#batch, time2, size).
            value(Tensor): 
                Value tensor (#batch, time2, size).
            pos_emb(Tensor): 
                Positional embedding tensor (#batch, 2*time1-1, size).
            mask(Tensor): 
                Mask tensor (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        # (batch, time1, head, d_k)
        q = q.transpose([0, 2, 1, 3])

        n_batch_pos = paddle.shape(pos_emb)[0]
        p = self.linear_pos(pos_emb).reshape(
            [n_batch_pos, -1, self.h, self.d_k])
        # (batch, head, 2*time1-1, d_k)
        p = p.transpose([0, 2, 1, 3])
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
        # (batch, head, time1, 2*time1-1)
        matrix_bd = paddle.matmul(q_with_bias_v, p.transpose([0, 1, 3, 2]))
        matrix_bd = self.rel_shift(matrix_bd)
        # (batch, head, time1, time2)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)


class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): 
            The number of heads.
        n_feat (int): 
            The number of features.
        dropout_rate (float): 
            Dropout rate.
        zero_triu (bool): 
            Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias_attr=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3

        self.pos_bias_u = paddle.create_parameter(
            shape=(self.h, self.d_k),
            dtype='float32',
            default_initializer=paddle.nn.initializer.XavierUniform())
        self.pos_bias_v = paddle.create_parameter(
            shape=(self.h, self.d_k),
            dtype='float32',
            default_initializer=paddle.nn.initializer.XavierUniform())

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x(Tensor): 
                Input tensor (batch, head, time1, time2).
        Returns:
            Tensor:Output tensor.
        """
        b, h, t1, t2 = paddle.shape(x)
        zero_pad = paddle.zeros((b, h, t1, 1))
        x_padded = paddle.concat([zero_pad, x], axis=-1)
        x_padded = paddle.reshape(x_padded, [b, h, t2 + 1, t1])
        # only keep the positions from 0 to time2
        x = paddle.reshape(x_padded[:, :, 1:], [b, h, t1, t2])

        if self.zero_triu:
            ones = paddle.ones((t1, t2))
            x = x * paddle.tril(ones, t2 - t1)[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query(Tensor): Query tensor (#batch, time1, size).
            key(Tensor): Key tensor (#batch, time2, size).
            value(Tensor): Value tensor (#batch, time2, size).
            pos_emb(Tensor): Positional embedding tensor (#batch, time1, size).
            mask(Tensor): Mask tensor (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        # (batch, time1, head, d_k)
        q = paddle.transpose(q, [0, 2, 1, 3])

        n_batch_pos = paddle.shape(pos_emb)[0]
        p = paddle.reshape(
            self.linear_pos(pos_emb), [n_batch_pos, -1, self.h, self.d_k])
        # (batch, head, time1, d_k)
        p = paddle.transpose(p, [0, 2, 1, 3])
        # (batch, head, time1, d_k)
        q_with_bias_u = paddle.transpose((q + self.pos_bias_u), [0, 2, 1, 3])
        # (batch, head, time1, d_k)
        q_with_bias_v = paddle.transpose((q + self.pos_bias_v), [0, 2, 1, 3])

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = paddle.matmul(q_with_bias_u,
                                  paddle.transpose(k, [0, 1, 3, 2]))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = paddle.matmul(q_with_bias_v,
                                  paddle.transpose(p, [0, 1, 3, 2]))
        matrix_bd = self.rel_shift(matrix_bd)
        # (batch, head, time1, time2)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)
