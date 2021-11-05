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

    Parameters
    ----------
    n_head : int
        The number of heads.
    n_feat : int
        The number of features.
    dropout_rate : float
        Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
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

        Parameters
        ----------
        query : paddle.Tensor
            query tensor (#batch, time1, size).
        key : paddle.Tensor
            Key tensor (#batch, time2, size).
        value : paddle.Tensor
            Value tensor (#batch, time2, size).

        Returns
        ----------
        paddle.Tensor
            Transformed query tensor (#batch, n_head, time1, d_k).
        paddle.Tensor
            Transformed key tensor (#batch, n_head, time2, d_k).
        paddle.Tensor
            Transformed value tensor (#batch, n_head, time2, d_k).
        """
        n_batch = query.shape[0]

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

        Parameters
        ----------
        value : paddle.Tensor
            Transformed value (#batch, n_head, time2, d_k).
        scores : paddle.Tensor
            Attention score (#batch, n_head, time1, time2).
        mask :  paddle.Tensor
            Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns
        ----------
        paddle.Tensor:
            Transformed value (#batch, time1, d_model)
            weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.shape[0]
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

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask=None):
        """Compute scaled dot product attention.

        Parameters
        ----------
        query : paddle.Tensor
            Query tensor (#batch, time1, size).
        key : paddle.Tensor
            Key tensor (#batch, time2, size).
        value : paddle.Tensor
            Value tensor (#batch, time2, size).
        mask : paddle.Tensor
            Mask tensor (#batch, 1, time2) or (#batch, time1, time2).

        Returns
        ----------
        paddle.Tensor
            Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = paddle.matmul(q, k.transpose(
            (0, 1, 3, 2))) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)
