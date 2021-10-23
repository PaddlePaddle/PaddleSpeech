# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import math

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F


def scaled_dot_product_attention(q, k, v, mask=None, dropout=0.0,
                                 training=True):
    r"""Scaled dot product attention with masking. 
    
    Assume that q, k, v all have the same leading dimensions (denoted as * in 
    descriptions below). Dropout is applied to attention weights before 
    weighted sum of values.

    Parameters
    -----------
    q : Tensor [shape=(\*, T_q, d)]
        the query tensor.
    k : Tensor [shape=(\*, T_k, d)]
        the key tensor.
    v : Tensor [shape=(\*, T_k, d_v)]
        the value tensor.
    mask : Tensor, [shape=(\*, T_q, T_k) or broadcastable shape], optional
        the mask tensor, zeros correspond to paddings. Defaults to None.

    Returns
    ----------
    out : Tensor [shape=(\*, T_q, d_v)]
        the context vector.
    attn_weights : Tensor [shape=(\*, T_q, T_k)]
        the attention weights.
    """
    d = q.shape[-1]  # we only support imperative execution
    qk = paddle.matmul(q, k, transpose_y=True)
    scaled_logit = paddle.scale(qk, 1.0 / math.sqrt(d))

    if mask is not None:
        scaled_logit += paddle.scale((1.0 - mask), -1e9)  # hard coded here

    attn_weights = F.softmax(scaled_logit, axis=-1)
    attn_weights = F.dropout(attn_weights, dropout, training=training)
    out = paddle.matmul(attn_weights, v)
    return out, attn_weights


def drop_head(x, drop_n_heads, training=True):
    """Drop n context vectors from multiple ones.

    Parameters
    ----------
    x : Tensor [shape=(batch_size, num_heads, time_steps, channels)]
        The input, multiple context vectors.
    drop_n_heads : int [0<= drop_n_heads <= num_heads]
        Number of vectors to drop.
    training : bool
        A flag indicating whether it is in training. If `False`, no dropout is 
        applied.

    Returns
    -------
    Tensor
        The output.
    """
    if not training or (drop_n_heads == 0):
        return x

    batch_size, num_heads, _, _ = x.shape
    # drop all heads
    if num_heads == drop_n_heads:
        return paddle.zeros_like(x)

    mask = np.ones([batch_size, num_heads])
    mask[:, :drop_n_heads] = 0
    for subarray in mask:
        np.random.shuffle(subarray)
    scale = float(num_heads) / (num_heads - drop_n_heads)
    mask = scale * np.reshape(mask, [batch_size, num_heads, 1, 1])
    out = x * paddle.to_tensor(mask)
    return out


def _split_heads(x, num_heads):
    batch_size, time_steps, _ = x.shape
    x = paddle.reshape(x, [batch_size, time_steps, num_heads, -1])
    x = paddle.transpose(x, [0, 2, 1, 3])
    return x


def _concat_heads(x):
    batch_size, _, time_steps, _ = x.shape
    x = paddle.transpose(x, [0, 2, 1, 3])
    x = paddle.reshape(x, [batch_size, time_steps, -1])
    return x


# Standard implementations of Monohead Attention & Multihead Attention
class MonoheadAttention(nn.Layer):
    """Monohead Attention module.

    Parameters
    ----------
    model_dim : int
        Feature size of the query.
    dropout : float, optional
        Dropout probability of scaled dot product attention and final context
        vector. Defaults to 0.0.
    k_dim : int, optional
        Feature size of the key of each scaled dot product attention. If not
        provided, it is set to `model_dim / num_heads`. Defaults to None.
    v_dim : int, optional
        Feature size of the key of each scaled dot product attention. If not
        provided, it is set to `model_dim / num_heads`. Defaults to None.
    """

    def __init__(self,
                 model_dim: int,
                 dropout: float=0.0,
                 k_dim: int=None,
                 v_dim: int=None):
        super(MonoheadAttention, self).__init__()
        k_dim = k_dim or model_dim
        v_dim = v_dim or model_dim
        self.affine_q = nn.Linear(model_dim, k_dim)
        self.affine_k = nn.Linear(model_dim, k_dim)
        self.affine_v = nn.Linear(model_dim, v_dim)
        self.affine_o = nn.Linear(v_dim, model_dim)

        self.model_dim = model_dim
        self.dropout = dropout

    def forward(self, q, k, v, mask):
        """Compute context vector and attention weights.
        
        Parameters
        -----------
        q : Tensor [shape=(batch_size, time_steps_q, model_dim)]
            The queries.
        k : Tensor [shape=(batch_size, time_steps_k, model_dim)]
            The keys.
        v : Tensor [shape=(batch_size, time_steps_k, model_dim)]
            The values.
        mask : Tensor [shape=(batch_size, times_steps_q, time_steps_k] or broadcastable shape
            The mask.

        Returns
        ----------
        out : Tensor [shape=(batch_size, time_steps_q, model_dim)]
            The context vector.
        attention_weights : Tensor [shape=(batch_size, times_steps_q, time_steps_k)]
            The attention weights.
        """
        q = self.affine_q(q)  # (B, T, C)
        k = self.affine_k(k)
        v = self.affine_v(v)

        context_vectors, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, self.dropout, self.training)

        out = self.affine_o(context_vectors)
        return out, attention_weights


class MultiheadAttention(nn.Layer):
    """Multihead Attention module.

    Parameters
    -----------
    model_dim: int
        The feature size of query.
    num_heads : int
        The number of attention heads.
    dropout : float, optional
        Dropout probability of scaled dot product attention and final context
        vector. Defaults to 0.0.
    k_dim : int, optional
        Feature size of the key of each scaled dot product attention. If not
        provided, it is set to ``model_dim / num_heads``. Defaults to None.
    v_dim : int, optional
        Feature size of the key of each scaled dot product attention. If not
        provided, it is set to ``model_dim / num_heads``. Defaults to None.

    Raises
    ---------
    ValueError
        If ``model_dim`` is not divisible by ``num_heads``.
    """

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout: float=0.0,
                 k_dim: int=None,
                 v_dim: int=None):
        super(MultiheadAttention, self).__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        depth = model_dim // num_heads
        k_dim = k_dim or depth
        v_dim = v_dim or depth
        self.affine_q = nn.Linear(model_dim, num_heads * k_dim)
        self.affine_k = nn.Linear(model_dim, num_heads * k_dim)
        self.affine_v = nn.Linear(model_dim, num_heads * v_dim)
        self.affine_o = nn.Linear(num_heads * v_dim, model_dim)

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dropout = dropout

    def forward(self, q, k, v, mask):
        """Compute context vector and attention weights.
        
        Parameters
        -----------
        q : Tensor [shape=(batch_size, time_steps_q, model_dim)]
            The queries.
        k : Tensor [shape=(batch_size, time_steps_k, model_dim)]
            The keys.
        v : Tensor [shape=(batch_size, time_steps_k, model_dim)]
            The values.
        mask : Tensor [shape=(batch_size, times_steps_q, time_steps_k] or broadcastable shape
            The mask.

        Returns
        ----------
        out : Tensor [shape=(batch_size, time_steps_q, model_dim)]
            The context vector.
        attention_weights : Tensor [shape=(batch_size, times_steps_q, time_steps_k)]
            The attention weights.
        """
        q = _split_heads(self.affine_q(q), self.num_heads)  # (B, h, T, C)
        k = _split_heads(self.affine_k(k), self.num_heads)
        v = _split_heads(self.affine_v(v), self.num_heads)
        mask = paddle.unsqueeze(mask, 1)  # unsqueeze for the h dim

        context_vectors, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, self.dropout, self.training)
        # NOTE: there is more sophisticated implementation: Scheduled DropHead
        context_vectors = _concat_heads(context_vectors)  # (B, T, h*C)
        out = self.affine_o(context_vectors)
        return out, attention_weights


class LocationSensitiveAttention(nn.Layer):
    """Location Sensitive Attention module.

    Reference: `Attention-Based Models for Speech Recognition <https://arxiv.org/pdf/1506.07503.pdf>`_

    Parameters
    -----------
    d_query: int
        The feature size of query.
    d_key : int
        The feature size of key.
    d_attention : int
        The feature size of dimension.
    location_filters : int
        Filter size of attention convolution.
    location_kernel_size : int
        Kernel size of attention convolution.
    """

    def __init__(self,
                 d_query: int,
                 d_key: int,
                 d_attention: int,
                 location_filters: int,
                 location_kernel_size: int):
        super().__init__()

        self.query_layer = nn.Linear(d_query, d_attention, bias_attr=False)
        self.key_layer = nn.Linear(d_key, d_attention, bias_attr=False)
        self.value = nn.Linear(d_attention, 1, bias_attr=False)

        # Location Layer
        self.location_conv = nn.Conv1D(
            2,
            location_filters,
            kernel_size=location_kernel_size,
            padding=int((location_kernel_size - 1) / 2),
            bias_attr=False,
            data_format='NLC')
        self.location_layer = nn.Linear(
            location_filters, d_attention, bias_attr=False)

    def forward(self,
                query,
                processed_key,
                value,
                attention_weights_cat,
                mask=None):
        """Compute context vector and attention weights.
        
        Parameters
        -----------
        query : Tensor [shape=(batch_size, d_query)]
            The queries.
        processed_key : Tensor [shape=(batch_size, time_steps_k, d_attention)]
            The keys after linear layer.
        value : Tensor [shape=(batch_size, time_steps_k, d_key)]
            The values.
        attention_weights_cat : Tensor [shape=(batch_size, time_step_k, 2)]
            Attention weights concat.
        mask : Tensor, optional
            The mask. Shape should be (batch_size, times_steps_k, 1).
            Defaults to None.

        Returns
        ----------
        attention_context : Tensor [shape=(batch_size, d_attention)]
            The context vector.
        attention_weights : Tensor [shape=(batch_size, time_steps_k)]
            The attention weights.
        """

        processed_query = self.query_layer(paddle.unsqueeze(query, axis=[1]))
        processed_attention_weights = self.location_layer(
            self.location_conv(attention_weights_cat))
        # (B, T_enc, 1)
        alignment = self.value(
            paddle.tanh(processed_attention_weights + processed_key +
                        processed_query))

        if mask is not None:
            alignment = alignment + (1.0 - mask) * -1e9

        attention_weights = F.softmax(alignment, axis=1)
        attention_context = paddle.matmul(
            attention_weights, value, transpose_x=True)

        attention_weights = paddle.squeeze(attention_weights, axis=-1)
        attention_context = paddle.squeeze(attention_context, axis=1)

        return attention_context, attention_weights
