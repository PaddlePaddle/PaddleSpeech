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
from typing import List
from typing import Tuple

import paddle
from paddle import nn
from paddle.nn import initializer as I

from paddlespeech.s2t.modules.align import Linear
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = [
    "MultiHeadedAttention", "RelPositionMultiHeadedAttention",
    "RoPERelPositionMultiHeadedAttention"
]

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
        self.n_feat = n_feat
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

        q = self.linear_q(query).reshape([n_batch, -1, self.h, self.d_k])
        k = self.linear_k(key).reshape([n_batch, -1, self.h, self.d_k])
        v = self.linear_v(value).reshape([n_batch, -1, self.h, self.d_k])

        q = q.transpose([0, 2, 1, 3])  # (batch, head, time1, d_k)
        k = k.transpose([0, 2, 1, 3])  # (batch, head, time2, d_k)
        v = v.transpose([0, 2, 1, 3])  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
            self,
            value: paddle.Tensor,
            scores: paddle.Tensor,
            mask: paddle.Tensor=paddle.ones([0, 0, 0], dtype=paddle.bool)
    ) -> paddle.Tensor:
        """Compute attention context vector.
        Args:
            value (paddle.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (paddle.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (paddle.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
        Returns:
            paddle.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.shape[0]

        # When `if mask.size(2) > 0` be True:
        # 1. training.
        # 2. oonx(16/4, chunk_size/history_size), feed real cache and real mask for the 1st chunk.
        # When will `if mask.size(2) > 0` be False?
        # 1. onnx(16/-1, -1/-1, 16/0)
        # 2. jit (16/-1, -1/-1, 16/0, 16/4)
        if mask.shape[2] > 0:  # time2 > 0
            mask = mask.unsqueeze(1).equal(0)  # (batch, 1, *, time2)
            # for last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.shape[-1]]
            scores = scores.masked_fill(mask, -float('inf'))
            attn = paddle.softmax(
                scores, axis=-1).masked_fill(mask,
                                             0.0)  # (batch, head, time1, time2)
        else:
            attn = paddle.softmax(
                scores, axis=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = paddle.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose([0, 2, 1, 3]).reshape(
            [n_batch, -1, self.h * self.d_k])  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self,
                query: paddle.Tensor,
                key: paddle.Tensor,
                value: paddle.Tensor,
                mask: paddle.Tensor=paddle.ones([0, 0, 0], dtype=paddle.bool),
                pos_emb: paddle.Tensor=paddle.empty([0]),
                cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0])
                ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute scaled dot product attention.
       Args:
            query (paddle.Tensor): Query tensor (#batch, time1, size).
            key (paddle.Tensor): Key tensor (#batch, time2, size).
            value (paddle.Tensor): Value tensor (#batch, time2, size).
            mask (paddle.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (paddle.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            paddle.Tensor: Output tensor (#batch, time1, d_model).
            paddle.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        # (B,T,D) -> (B,T,H,D/H)
        q, k, v = self.forward_qkv(query, key, value)

        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.shape[0] > 0:
            # last dim `d_k * 2` for (key, val)
            key_cache, value_cache = paddle.split(cache, 2, axis=-1)
            k = paddle.concat([key_cache, k], axis=2)
            v = paddle.concat([value_cache, v], axis=2)
        # We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = paddle.concat((k, v), axis=-1)

        # scores = paddle.matmul(q,
        #    k.transpose([0, 1, 3, 2])) / math.sqrt(self.d_k)
        scores = paddle.matmul(q, k, transpose_y=True) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding."""

    def __init__(self,
                 n_head,
                 n_feat,
                 dropout_rate,
                 adaptive_scale=False,
                 init_weights=False):
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
        self.adaptive_scale = adaptive_scale
        if self.adaptive_scale:
            ada_scale = self.create_parameter(
                [1, 1, n_feat], default_initializer=I.Constant(1.0))
            self.add_parameter('ada_scale', ada_scale)
            ada_bias = self.create_parameter(
                [1, 1, n_feat], default_initializer=I.Constant(0.0))
            self.add_parameter('ada_bias', ada_bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        input_max = (self.h * self.d_k)**-0.5
        self.linear_q._param_attr = paddle.nn.initializer.Uniform(
            low=-input_max, high=input_max)
        self.linear_q._bias_attr = paddle.nn.initializer.Uniform(
            low=-input_max, high=input_max)
        self.linear_k._param_attr = paddle.nn.initializer.Uniform(
            low=-input_max, high=input_max)
        self.linear_k._bias_attr = paddle.nn.initializer.Uniform(
            low=-input_max, high=input_max)
        self.linear_v._param_attr = paddle.nn.initializer.Uniform(
            low=-input_max, high=input_max)
        self.linear_v._bias_attr = paddle.nn.initializer.Uniform(
            low=-input_max, high=input_max)
        self.linear_pos._param_attr = paddle.nn.initializer.Uniform(
            low=-input_max, high=input_max)
        self.linear_pos._bias_attr = paddle.nn.initializer.Uniform(
            low=-input_max, high=input_max)
        self.linear_out._param_attr = paddle.nn.initializer.Uniform(
            low=-input_max, high=input_max)
        self.linear_out._bias_attr = paddle.nn.initializer.Uniform(
            low=-input_max, high=input_max)

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

        x_padded = x_padded.reshape(
            [x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2]])
        x = x_padded[:, :, 1:].view_as(x)  # [B, H, T1, T1]

        if zero_triu:
            ones = paddle.ones((x.shape[2], x.shape[3]))
            x = x * paddle.tril(ones, x.shape[3] - x.shape[2])[None, None, :, :]

        return x

    def forward(self,
                query: paddle.Tensor,
                key: paddle.Tensor,
                value: paddle.Tensor,
                mask: paddle.Tensor=paddle.ones([0, 0, 0], dtype=paddle.bool),
                pos_emb: paddle.Tensor=paddle.empty([0]),
                cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0])
                ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (paddle.Tensor): Query tensor (#batch, time1, size).
            key (paddle.Tensor): Key tensor (#batch, time2, size).
            value (paddle.Tensor): Value tensor (#batch, time2, size).
            mask (paddle.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (paddle.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (paddle.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            paddle.Tensor: Output tensor (#batch, time1, d_model).
            paddle.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        if self.adaptive_scale:
            query = self.ada_scale * query + self.ada_bias
            key = self.ada_scale * key + self.ada_bias
            value = self.ada_scale * value + self.ada_bias

        q, k, v = self.forward_qkv(query, key, value)
        # q = q.transpose([0, 2, 1, 3])  # (batch, time1, head, d_k)

        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.shape[0] > 0:
            # last dim `d_k * 2` for (key, val)
            key_cache, value_cache = paddle.split(cache, 2, axis=-1)
            k = paddle.concat([key_cache, k], axis=2)
            v = paddle.concat([value_cache, v], axis=2)
        # We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = paddle.concat((k, v), axis=-1)

        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb).reshape(
            [n_batch_pos, -1, self.h, self.d_k])
        p = p.transpose([0, 2, 1, 3])  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        # q_with_bias_u = (q + self.pos_bias_u).transpose([0, 2, 1, 3])
        q_with_bias_u = q + self.pos_bias_u.unsqueeze(1)
        # (batch, head, time1, d_k)
        # q_with_bias_v = (q + self.pos_bias_v).transpose([0, 2, 1, 3])
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(1)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        # matrix_ac = paddle.matmul(q_with_bias_u, k.transpose([0, 1, 3, 2]))
        matrix_ac = paddle.matmul(q_with_bias_u, k, transpose_y=True)

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        # matrix_bd = paddle.matmul(q_with_bias_v, p.transpose([0, 1, 3, 2]))
        matrix_bd = paddle.matmul(q_with_bias_v, p, transpose_y=True)
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), new_cache


class RoPERelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with RoPE relative position encoding."""

    def __init__(self,
                 n_head,
                 n_feat,
                 dropout_rate,
                 adaptive_scale=False,
                 init_weights=False):
        """Construct an RelPositionMultiHeadedAttention object.
        Paper: https://arxiv.org/abs/1901.02860
        Args:
            n_head (int): The number of heads.
            n_feat (int): The number of features.
            dropout_rate (float): Dropout rate.
        """
        super().__init__(n_head, n_feat, dropout_rate)

    def align(self, tensor: paddle.Tensor, axes: List[int], ndim=None):
        """重新对齐tensor（批量版expand_dims）
        axes：原来的第i维对齐新tensor的第axes[i]维；
        ndim：新tensor的维度。
        """
        assert len(axes) == tensor.dim()
        assert ndim or min(axes) >= 0

        ndim = ndim or max(axes) + 1

        # a[0, None, 1] = a[0, np.newaxis, 1]
        indices = [None] * ndim
        for i in axes:
            # slice nothing, a[0, slice(None), 1] = a[0, :, 1]
            indices[i] = slice(None)

        return tensor[indices]

    def apply_rotary_position_embeddings(self, sinusoidal, *tensors):
        """应用RoPE到tensors中
        其中，sinusoidal.shape=[B, T, D]，tensors为tensor的列表，而
        tensor.shape=[B, T, ..., D], or (B,H,T,D/H)
        """
        assert len(tensors) > 0, 'at least one input tensor'
        assert all(
            [tensor.shape == tensors[0].shape
             for tensor in tensors[1:]]), 'all tensors must have the same shape'

        # (B,H,T,D)
        ndim = tensors[0].dim()
        _, H, T, D = tensors[0].shape

        # sinusoidal shape same with tensors[0]
        # [B,T,D] -> [B,T,H,D/H] -> (B,H,T,D/H)
        # sinusoidal = self.align(sinusoidal, [0, 1, -1], ndim)
        sinusoidal = sinusoidal.reshape((1, T, H, D)).transpose([0, 2, 1, 3])

        # http://man.hubwiz.com/docset/TensorFlow.docset/Contents/Resources/Documents/api_docs/python/tf/keras/backend/repeat_elements.html
        # like np.repeat, x (s1, s2, s3), axis 1, (s1, s2*rep, s3)
        # [b,T, ..., d/2] -> [b,T, ..., d]
        cos_pos = paddle.repeat_interleave(sinusoidal[..., 1::2], 2, axis=-1)
        sin_pos = paddle.repeat_interleave(sinusoidal[..., 0::2], 2, axis=-1)
        outputs = []
        for tensor in tensors:
            # x2 = [-x2, x1, -x4, x3, ..., -x_d, x_{d-1}]
            tensor2 = paddle.stack([-tensor[..., 1::2], tensor[..., ::2]], ndim)
            tensor2 = paddle.reshape(tensor2, paddle.shape(tensor))

            # 公式 34, out = x * cos_pos + x2 * sin_pos
            outputs.append(tensor * cos_pos + tensor2 * sin_pos)
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self,
                query: paddle.Tensor,
                key: paddle.Tensor,
                value: paddle.Tensor,
                mask: paddle.Tensor=paddle.ones([0, 0, 0], dtype=paddle.bool),
                pos_emb: paddle.Tensor=paddle.empty([0]),
                cache: paddle.Tensor=paddle.zeros([0, 0, 0, 0])
                ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Ref: https://github.com/facebookresearch/llama/blob/main/llama/model.py
        Args:
            query (paddle.Tensor): Query tensor (#batch, time1, size).
            key (paddle.Tensor): Key tensor (#batch, time2, size).
            value (paddle.Tensor): Value tensor (#batch, time2, size).
            mask (paddle.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (paddle.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (paddle.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            paddle.Tensor: Output tensor (#batch, time1, d_model).
            paddle.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        # q = q.transpose([0, 2, 1, 3])  # (batch, time1, head, d_k)

        # f{q,k}(x_m, m) = R^d_{\theta, m} W_{q,k} x_m, m is position index
        # q_t always is chunk_size
        q_t = q.shape[2]
        q = self.apply_rotary_position_embeddings(pos_emb[:, -q_t:, :], q)
        # k will increase when in streaming decoding.
        k = self.apply_rotary_position_embeddings(pos_emb[:, -q_t:, :], k)

        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.shape[0] > 0:
            # last dim `d_k * 2` for (key, val)
            key_cache, value_cache = paddle.split(cache, 2, axis=-1)
            k = paddle.concat([key_cache, k], axis=2)
            v = paddle.concat([value_cache, v], axis=2)
        # We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = paddle.concat((k, v), axis=-1)

        # dot(q, k)
        scores = paddle.matmul(q, k, transpose_y=True) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache
