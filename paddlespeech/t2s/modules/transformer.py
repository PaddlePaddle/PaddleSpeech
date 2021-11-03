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
from paddle import nn
from paddle.nn import functional as F

from paddlespeech.t2s.modules import attention as attn

__all__ = [
    "PositionwiseFFN",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
]


class PositionwiseFFN(nn.Layer):
    """A faithful implementation of Position-wise Feed-Forward Network 
    in `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_.
    It is basically a 2-layer MLP, with relu actication and dropout in between.

    Parameters
    ----------
    input_size: int
        The feature size of the intput. It is also the feature size of the
        output.
    hidden_size: int
        The hidden size.
    dropout: float
        The probability of the Dropout applied to the output of the first
        layer, by default 0.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout=0.0):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)

        self.input_size = input_size
        self.hidden_szie = hidden_size

    def forward(self, x):
        r"""Forward pass of positionwise feed forward network.

        Parameters
        ----------
        x : Tensor [shape=(\*, input_size)]
            The input tensor, where ``\*`` means arbitary shape.

        Returns
        -------
        Tensor [shape=(\*, input_size)]
            The output tensor.
        """
        l1 = self.dropout(F.relu(self.linear1(x)))
        l2 = self.linear2(l1)
        return l2


class TransformerEncoderLayer(nn.Layer):
    """A faithful implementation of Transformer encoder layer in
    `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Parameters
    ----------
    d_model :int 
        The feature size of the input. It is also the feature size of the
        output.
    n_heads : int
        The number of heads of self attention (a ``MultiheadAttention``
        layer).
    d_ffn : int 
        The hidden size of the positional feed forward network (a
        ``PositionwiseFFN`` layer).
    dropout : float, optional
        The probability of the dropout in MultiHeadAttention and
        PositionwiseFFN, by default 0.

    Notes
    ------
    It uses the PostLN (post layer norm) scheme.
    """

    def __init__(self, d_model, n_heads, d_ffn, dropout=0.):
        super(TransformerEncoderLayer, self).__init__()
        self.self_mha = attn.MultiheadAttention(d_model, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm([d_model], epsilon=1e-6)

        self.ffn = PositionwiseFFN(d_model, d_ffn, dropout)
        self.layer_norm2 = nn.LayerNorm([d_model], epsilon=1e-6)

        self.dropout = dropout

    def forward(self, x, mask):
        """Forward pass of TransformerEncoderLayer.

        Parameters
        ----------
        x : Tensor [shape=(batch_size, time_steps, d_model)]
            The input.
        mask : Tensor
            The padding mask. The shape is (batch_size, time_steps,
            time_steps) or broadcastable shape.

        Returns
        -------
        x :Tensor [shape=(batch_size, time_steps, d_model)]
            The encoded output.

        attn_weights : Tensor [shape=(batch_size, n_heads, time_steps, time_steps)]
            The attention weights of the self attention.
        """
        context_vector, attn_weights = self.self_mha(x, x, x, mask)
        x = self.layer_norm1(
            F.dropout(x + context_vector, self.dropout, training=self.training))

        x = self.layer_norm2(
            F.dropout(x + self.ffn(x), self.dropout, training=self.training))
        return x, attn_weights


class TransformerDecoderLayer(nn.Layer):
    """A faithful implementation of Transformer decoder layer in 
    `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Parameters
    ----------
    d_model :int 
        The feature size of the input. It is also the feature size of the
        output.
    n_heads : int
        The number of heads of attentions (``MultiheadAttention``
        layers).
    d_ffn : int 
        The hidden size of the positional feed forward network (a
        ``PositionwiseFFN`` layer).
    dropout : float, optional
        The probability of the dropout in MultiHeadAttention and
        PositionwiseFFN, by default 0.

    Notes
    ------
    It uses the PostLN (post layer norm) scheme.
    """

    def __init__(self, d_model, n_heads, d_ffn, dropout=0.):
        super(TransformerDecoderLayer, self).__init__()
        self.self_mha = attn.MultiheadAttention(d_model, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm([d_model], epsilon=1e-6)

        self.cross_mha = attn.MultiheadAttention(d_model, n_heads, dropout)
        self.layer_norm2 = nn.LayerNorm([d_model], epsilon=1e-6)

        self.ffn = PositionwiseFFN(d_model, d_ffn, dropout)
        self.layer_norm3 = nn.LayerNorm([d_model], epsilon=1e-6)

        self.dropout = dropout

    def forward(self, q, k, v, encoder_mask, decoder_mask):
        """Forward pass of TransformerEncoderLayer.

        Parameters
        ----------
        q : Tensor [shape=(batch_size, time_steps_q, d_model)]
            The decoder input.
        k : Tensor [shape=(batch_size, time_steps_k, d_model)]
            The keys.
        v : Tensor [shape=(batch_size, time_steps_k, d_model)]
            The values
        encoder_mask : Tensor
            Encoder padding mask, shape is ``(batch_size, time_steps_k,
            time_steps_k)`` or broadcastable shape.
        decoder_mask : Tensor
            Decoder mask, shape is ``(batch_size, time_steps_q, time_steps_k)``
            or broadcastable shape. 

        Returns
        --------
        q : Tensor [shape=(batch_size, time_steps_q, d_model)]
            The decoder output.
        self_attn_weights : Tensor [shape=(batch_size, n_heads, time_steps_q, time_steps_q)]
            Decoder self attention.

        cross_attn_weights : Tensor [shape=(batch_size, n_heads, time_steps_q, time_steps_k)]
            Decoder-encoder cross attention.
        """
        context_vector, self_attn_weights = self.self_mha(q, q, q, decoder_mask)
        q = self.layer_norm1(
            F.dropout(q + context_vector, self.dropout, training=self.training))

        context_vector, cross_attn_weights = self.cross_mha(q, k, v,
                                                            encoder_mask)
        q = self.layer_norm2(
            F.dropout(q + context_vector, self.dropout, training=self.training))

        q = self.layer_norm3(
            F.dropout(q + self.ffn(q), self.dropout, training=self.training))
        return q, self_attn_weights, cross_attn_weights
