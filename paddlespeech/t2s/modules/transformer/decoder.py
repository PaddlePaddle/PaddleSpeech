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
# 暂时删除了 dyminic conv
"""Decoder definition."""
import logging
from typing import Any
from typing import List
from typing import Tuple

import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.t2s.modules.layer_norm import LayerNorm
from paddlespeech.t2s.modules.transformer.attention import MultiHeadedAttention
from paddlespeech.t2s.modules.transformer.decoder_layer import DecoderLayer
from paddlespeech.t2s.modules.transformer.embedding import PositionalEncoding
from paddlespeech.t2s.modules.transformer.lightconv import LightweightConvolution
from paddlespeech.t2s.modules.transformer.mask import subsequent_mask
from paddlespeech.t2s.modules.transformer.positionwise_feed_forward import PositionwiseFeedForward
from paddlespeech.t2s.modules.transformer.repeat import repeat


class Decoder(nn.Layer):
    """Transfomer decoder module.

    Args:
        odim (int): 
            Output diminsion.
        self_attention_layer_type (str): 
            Self-attention layer type.
        attention_dim (int): 
            Dimention of attention.
        attention_heads (int): 
            The number of heads of multi head attention.
        conv_wshare (int):
            The number of kernel of convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        conv_kernel_length (Union[int, str]):
            Kernel size str of convolution
            (e.g. 71_71_71_71_71_71). Only used in self_attention_layer_type == "lightconv*" or "dynamiconv*".
        conv_usebias (bool): 
            Whether to use bias in convolution. Only used in
            self_attention_layer_type == "lightconv*" or "dynamiconv*".
        linear_units(int): 
            The number of units of position-wise feed forward.
        num_blocks (int): 
            The number of decoder blocks.
        dropout_rate (float): 
            Dropout rate.
        positional_dropout_rate (float): 
            Dropout rate after adding positional encoding.
        self_attention_dropout_rate (float): 
            Dropout rate in self-attention.
        src_attention_dropout_rate (float): 
            Dropout rate in source-attention.
        input_layer (Union[str, nn.Layer]): 
            Input layer type.
        use_output_layer (bool): 
            Whether to use output layer.
        pos_enc_class (nn.Layer): 
            Positional encoding module class.
            `PositionalEncoding `or `ScaledPositionalEncoding`
        normalize_before (bool): 
            Whether to use layer_norm before the first block.
        concat_after (bool): 
            Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
            self,
            odim,
            selfattention_layer_type="selfattn",
            attention_dim=256,
            attention_heads=4,
            conv_wshare=4,
            conv_kernel_length=11,
            conv_usebias=False,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            self_attention_dropout_rate=0.0,
            src_attention_dropout_rate=0.0,
            input_layer="embed",
            use_output_layer=True,
            pos_enc_class=PositionalEncoding,
            normalize_before=True,
            concat_after=False, ):
        """Construct an Decoder object."""
        nn.Layer.__init__(self)
        if input_layer == "embed":
            self.embed = nn.Sequential(
                nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif input_layer == "linear":
            self.embed = nn.Sequential(
                nn.Linear(odim, attention_dim),
                nn.LayerNorm(attention_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate), )
        elif isinstance(input_layer, nn.Layer):
            self.embed = nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate))
        else:
            raise NotImplementedError("only `embed` or nn.Layer is supported.")
        self.normalize_before = normalize_before

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            logging.info("decoder self-attention layer type = self-attention")
            decoder_selfattn_layer = MultiHeadedAttention
            decoder_selfattn_layer_args = [
                (attention_heads, attention_dim, self_attention_dropout_rate, )
            ] * num_blocks
        elif selfattention_layer_type == "lightconv":
            logging.info(
                "decoder self-attention layer type = lightweight convolution")
            decoder_selfattn_layer = LightweightConvolution
            decoder_selfattn_layer_args = [(
                conv_wshare, attention_dim, self_attention_dropout_rate,
                int(conv_kernel_length.split("_")[lnum]), True, conv_usebias, )
                                           for lnum in range(num_blocks)]

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                decoder_selfattn_layer(*decoder_selfattn_layer_args[lnum]),
                MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after, ), )
        self.selfattention_layer_type = selfattention_layer_type
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = nn.Linear(attention_dim, odim)
        else:
            self.output_layer = None

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward decoder.
        Args:
            tgt(Tensor): 
                Input token ids, int64 (#batch, maxlen_out) if input_layer == "embed".
                In the other case, input tensor (#batch, maxlen_out, odim).
            tgt_mask(Tensor): 
                Input token mask (#batch, maxlen_out).
            memory(Tensor): 
                Encoded memory, float32 (#batch, maxlen_in, feat).
            memory_mask(Tensor): 
                Encoded memory mask (#batch, maxlen_in).

        Returns:
            Tensor:
                Decoded token score before softmax (#batch, maxlen_out, odim) if use_output_layer is True. 
                In the other case,final block outputs (#batch, maxlen_out, attention_dim).
            Tensor: 
                Score mask before softmax (#batch, maxlen_out).

        """
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(x, tgt_mask, memory,
                                                         memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, cache=None):
        """Forward one step.

        Args:
            tgt(Tensor): 
                Input token ids, int64 (#batch, maxlen_out).
            tgt_mask(Tensor): 
                Input token mask (#batch, maxlen_out).
            memory(Tensor): 
                Encoded memory, float32 (#batch, maxlen_in, feat).
            cache((List[Tensor]), optional): 
                List of cached tensors. (Default value = None)

        Returns:
            Tensor: 
                Output tensor (batch, maxlen_out, odim).
            List[Tensor]: 
                List of cache tensors of each decoder layer.

        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c)
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = F.log_softmax(self.output_layer(y), axis=-1)

        return y, new_cache

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys)).unsqueeze(0)
        if self.selfattention_layer_type != "selfattn":
            # TODO(karita): implement cache
            logging.warning(
                f"{self.selfattention_layer_type} does not support cached decoding."
            )
            state = None
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state)
        return logp.squeeze(0), state

    # batch beam search API (see BatchScorerInterface)
    def batch_score(self,
                    ys: paddle.Tensor,
                    states: List[Any],
                    xs: paddle.Tensor) -> Tuple[paddle.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys(Tensor): 
                paddle.int64 prefix tokens (n_batch, ylen).
            states(List[Any]): 
                Scorer states for prefix tokens.
            xs(Tensor): 
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[Tensor, List[Any]]:
                Tuple ofbatchfied scores for next token with shape of `(n_batch, n_vocab)` and next state list for ys.

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
        ys_mask = subsequent_mask(ys.shape[-1]).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)]
                      for b in range(n_batch)]
        return logp, state_list
