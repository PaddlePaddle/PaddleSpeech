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
"""Decoder definition."""
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import paddle
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.s2t.decoders.scorers.scorer_interface import BatchScorerInterface
from paddlespeech.s2t.modules.align import Embedding
from paddlespeech.s2t.modules.align import LayerNorm
from paddlespeech.s2t.modules.align import Linear
from paddlespeech.s2t.modules.attention import MultiHeadedAttention
from paddlespeech.s2t.modules.decoder_layer import DecoderLayer
from paddlespeech.s2t.modules.embedding import PositionalEncoding
from paddlespeech.s2t.modules.mask import make_non_pad_mask
from paddlespeech.s2t.modules.mask import make_xs_mask
from paddlespeech.s2t.modules.mask import subsequent_mask
from paddlespeech.s2t.modules.positionwise_feed_forward import PositionwiseFeedForward
from paddlespeech.s2t.utils.log import Log
logger = Log(__name__).getlog()

__all__ = ["TransformerDecoder"]


class TransformerDecoder(BatchScorerInterface, nn.Layer):
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

    def __init__(self,
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
                 concat_after: bool=False,
                 max_len: int=5000):

        assert check_argument_types()

        nn.Layer.__init__(self)
        self.selfattention_layer_type = 'selfattn'
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = nn.Sequential(
                Embedding(vocab_size, attention_dim),
                PositionalEncoding(
                    attention_dim, positional_dropout_rate, max_len=max_len), )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = LayerNorm(attention_dim, epsilon=1e-12)
        self.use_output_layer = use_output_layer
        self.output_layer = Linear(attention_dim, vocab_size)

        self.decoders = nn.LayerList([
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

    def forward(self,
                memory: paddle.Tensor,
                memory_mask: paddle.Tensor,
                ys_in_pad: paddle.Tensor,
                ys_in_lens: paddle.Tensor,
                r_ys_in_pad: paddle.Tensor=paddle.empty([0]),
                reverse_weight: float=0.0
                ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
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
        m = subsequent_mask(tgt_mask.shape[-1]).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        x, _ = self.embed(tgt)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return x, paddle.to_tensor(0.0), olens

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

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x):
        """Score.
        ys: (ylen,)
        x: (xlen, n_feat)
        """
        ys_mask = subsequent_mask(len(ys)).unsqueeze(0)  # (B,L,L)
        x_mask = make_xs_mask(x.unsqueeze(0)).unsqueeze(1)  # (B,1,T)
        if self.selfattention_layer_type != "selfattn":
            # TODO(karita): implement cache
            logging.warning(
                f"{self.selfattention_layer_type} does not support cached decoding."
            )
            state = None
        logp, state = self.forward_one_step(
            x.unsqueeze(0), x_mask, ys.unsqueeze(0), ys_mask, cache=state)
        return logp.squeeze(0), state

    # batch beam search API (see BatchScorerInterface)
    def batch_score(self,
                    ys: paddle.Tensor,
                    states: List[Any],
                    xs: paddle.Tensor) -> Tuple[paddle.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (paddle.Tensor): paddle.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (paddle.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[paddle.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                paddle.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.shape[-1]).unsqueeze(0)  # (B,L,L)
        xs_mask = make_xs_mask(xs).unsqueeze(1)  # (B,1,T)
        logp, states = self.forward_one_step(
            xs, xs_mask, ys, ys_mask, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)]
                      for b in range(n_batch)]
        return logp, state_list


class BiTransformerDecoder(BatchScorerInterface, nn.Layer):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(self,
                 vocab_size: int,
                 encoder_output_size: int,
                 attention_heads: int=4,
                 linear_units: int=2048,
                 num_blocks: int=6,
                 r_num_blocks: int=0,
                 dropout_rate: float=0.1,
                 positional_dropout_rate: float=0.1,
                 self_attention_dropout_rate: float=0.0,
                 src_attention_dropout_rate: float=0.0,
                 input_layer: str="embed",
                 use_output_layer: bool=True,
                 normalize_before: bool=True,
                 concat_after: bool=False,
                 max_len: int=5000):

        assert check_argument_types()

        nn.Layer.__init__(self)
        self.left_decoder = TransformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before, concat_after,
            max_len)

        self.right_decoder = TransformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            r_num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before, concat_after,
            max_len)

    def forward(
            self,
            memory: paddle.Tensor,
            memory_mask: paddle.Tensor,
            ys_in_pad: paddle.Tensor,
            ys_in_lens: paddle.Tensor,
            r_ys_in_pad: paddle.Tensor,
            reverse_weight: float=0.0,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        l_x, _, olens = self.left_decoder(memory, memory_mask, ys_in_pad,
                                          ys_in_lens)
        r_x = paddle.zeros([1])
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(memory, memory_mask, r_ys_in_pad,
                                               ys_in_lens)
        return l_x, r_x, olens

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
            y.shape` is (batch, maxlen_out, token)
        """
        return self.left_decoder.forward_one_step(memory, memory_mask, tgt,
                                                  tgt_mask, cache)
