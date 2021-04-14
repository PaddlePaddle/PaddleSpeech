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
"""Decoder definition."""
from typing import Tuple, List, Optional
from typeguard import check_argument_types
import logging

import paddle
from paddle import nn

from deepspeech.modules.attention import MultiHeadedAttention
from deepspeech.modules.decoder_layer import DecoderLayer
from deepspeech.modules.embedding import PositionalEncoding
from deepspeech.modules.positionwise_feed_forward import PositionwiseFeedForward
from deepspeech.modules.mask import subsequent_mask
from deepspeech.modules.mask import make_non_pad_mask

logger = logging.getLogger(__name__)

__all__ = ["TransformerDecoder"]


class TransformerDecoder(nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type, `embed`
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding module
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(
            self,
            vocab_size: int,
            encoder_output_size: int,
            attention_heads: int=4,
            linear_units: int=2048,
            num_blocks: int=6,
            dropout_rate: float=0.1,
            positional_dropout_rate: float=0.1,
            self_attention_dropout_rate: float=0.0,
            src_attention_dropout_rate: float=0.0,
            input_layer: str="embed",
            use_output_layer: bool=True,
            normalize_before: bool=True,
            concat_after: bool=False, ):

        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = nn.Sequential(
                nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate), )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(attention_dim, epsilon=1e-12)
        self.use_output_layer = use_output_layer
        self.output_layer = nn.Linear(attention_dim, vocab_size)

        self.decoders = nn.ModuleList([
            DecoderLayer(
                size=attention_dim,
                self_attn=MultiHeadedAttention(attention_heads, attention_dim,
                                               self_attention_dropout_rate),
                src_attn=MultiHeadedAttention(attention_heads, attention_dim,
                                              src_attention_dropout_rate),
                feed_forward=PositionwiseFeedForward(
                    attention_dim, linear_units, dropout_rate),
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after, ) for _ in range(num_blocks)
        ])

    def forward(
            self,
            memory: paddle.Tensor,
            memory_mask: paddle.Tensor,
            ys_in_pad: paddle.Tensor,
            ys_in_lens: paddle.Tensor, ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (make_non_pad_mask(ys_in_lens).unsqueeze(1))
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1)).unsqueeze(0)
        # tgt_mask: (B, L, L)
        # TODO(Hui Zhang): not support & for tensor
        # tgt_mask = tgt_mask & m
        tgt_mask = tgt_mask.logical_and(m)

        x, _ = self.embed(tgt)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)

        # TODO(Hui Zhang): reduce_sum not support bool type
        # olens = tgt_mask.sum(1)
        olens = tgt_mask.astype(paddle.int).sum(1)
        return x, olens

    def forward_one_step(
            self,
            memory: paddle.Tensor,
            memory_mask: paddle.Tensor,
            tgt: paddle.Tensor,
            tgt_mask: paddle.Tensor,
            cache: Optional[List[paddle.Tensor]]=None,
    ) -> Tuple[paddle.Tensor, List[paddle.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out, maxlen_out)
                      dtype=paddle.bool
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
                y.shape` is (batch, token)
        """
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = paddle.log_softmax(self.output_layer(y), axis=-1)
        return y, new_cache
