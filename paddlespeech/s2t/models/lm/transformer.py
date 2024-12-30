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
from typing import Any
from typing import List
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlespeech.s2t.decoders.scorers.scorer_interface import BatchScorerInterface
from paddlespeech.s2t.models.lm_interface import LMInterface
from paddlespeech.s2t.modules.encoder import TransformerEncoder
from paddlespeech.s2t.modules.mask import subsequent_mask
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


class TransformerLM(nn.Layer, LMInterface, BatchScorerInterface):
    def __init__(self,
                 n_vocab: int,
                 pos_enc: str=None,
                 embed_unit: int=128,
                 att_unit: int=256,
                 head: int=2,
                 unit: int=1024,
                 layer: int=4,
                 dropout_rate: float=0.5,
                 emb_dropout_rate: float=0.0,
                 att_dropout_rate: float=0.0,
                 tie_weights: bool=False,
                 **kwargs):
        nn.Layer.__init__(self)

        if pos_enc == "sinusoidal":
            pos_enc_layer_type = "abs_pos"
        elif pos_enc is None:
            pos_enc_layer_type = "no_pos"
        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")

        self.embed = nn.Embedding(n_vocab, embed_unit)

        if emb_dropout_rate == 0.0:
            self.embed_drop = None
        else:
            self.embed_drop = nn.Dropout(emb_dropout_rate)

        self.encoder = TransformerEncoder(
            input_size=embed_unit,
            output_size=att_unit,
            attention_heads=head,
            linear_units=unit,
            num_blocks=layer,
            dropout_rate=dropout_rate,
            attention_dropout_rate=att_dropout_rate,
            input_layer="linear",
            pos_enc_layer_type=pos_enc_layer_type,
            concat_after=False,
            static_chunk_size=1,
            use_dynamic_chunk=False,
            use_dynamic_left_chunk=False)

        self.decoder = nn.Linear(att_unit, n_vocab)

        logger.info("Tie weights set to {}".format(tie_weights))
        logger.info("Dropout set to {}".format(dropout_rate))
        logger.info("Emb Dropout set to {}".format(emb_dropout_rate))
        logger.info("Att Dropout set to {}".format(att_dropout_rate))

        if tie_weights:
            assert (
                att_unit == embed_unit
            ), "Tie Weights: True need embedding and final dimensions to match"
            self.decoder.weight = self.embed.weight

    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(paddle.shape(ys_mask)[-1]).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, x: paddle.Tensor, t: paddle.Tensor
                ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute LM loss value from buffer sequences.

        Args:
            x (paddle.Tensor): Input ids. (batch, len)
            t (paddle.Tensor): Target ids. (batch, len)

        Returns:
            tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]: Tuple of
                loss to backward (scalar),
                negative log-likelihood of t: -log p(t) (scalar) and
                the number of elements in x (scalar)

        Notes:
            The last two return values are used
            in perplexity: p(t)^{-n} = exp(-log p(t) / n)

        """
        batch_size = paddle.shape(x)[0]
        xm = x != 0
        xlen = xm.sum(axis=1)
        if self.embed_drop is not None:
            emb = self.embed_drop(self.embed(x))
        else:
            emb = self.embed(x)
        h, _ = self.encoder(emb, xlen)
        y = self.decoder(h)
        loss = F.cross_entropy(
            y.reshape([-1, paddle.shape(y)[-1]]),
            t.reshape([-1]),
            reduction="none")
        mask = xm.to(loss.dtype)
        logp = loss * mask.reshape([-1])
        nll = logp.reshape([batch_size, -1]).sum(-1)
        nll_count = mask.sum(-1)
        logp = logp.sum()
        count = mask.sum()
        return logp / count, logp, count, nll, nll_count

    # beam search API (see ScorerInterface)
    def score(self, y: paddle.Tensor, state: Any,
              x: paddle.Tensor) -> Tuple[paddle.Tensor, Any]:
        """Score new token.

        Args:
            y (paddle.Tensor): 1D paddle.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (paddle.Tensor): encoder feature that generates ys.

        Returns:
            tuple[paddle.Tensor, Any]: Tuple of
                paddle.float32 scores for next token (n_vocab)
                and next state for ys

        """
        y = y.unsqueeze(0)

        if self.embed_drop is not None:
            emb = self.embed_drop(self.embed(y))
        else:
            emb = self.embed(y)

        h, _, cache = self.encoder.forward_one_step(
            emb, self._target_mask(y), cache=state)
        h = self.decoder(h[:, -1])
        logp = F.log_softmax(h).squeeze(0)
        return logp, cache

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
        n_layers = len(self.encoder.encoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                paddle.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        if self.embed_drop is not None:
            emb = self.embed_drop(self.embed(ys))
        else:
            emb = self.embed(ys)

        # batch decoding
        h, _, states = self.encoder.forward_one_step(
            emb, self._target_mask(ys), cache=batch_state)
        h = self.decoder(h[:, -1])
        logp = F.log_softmax(h)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)]
                      for b in range(n_batch)]
        return logp, state_list


if __name__ == "__main__":
    tlm = TransformerLM(
        n_vocab=5002,
        pos_enc=None,
        embed_unit=128,
        att_unit=512,
        head=8,
        unit=2048,
        layer=16,
        dropout_rate=0.5, )

    #     n_vocab: int,
    # pos_enc: str=None,
    # embed_unit: int=128,
    # att_unit: int=256,
    # head: int=2,
    # unit: int=1024,
    # layer: int=4,
    # dropout_rate: float=0.5,
    # emb_dropout_rate: float = 0.0,
    # att_dropout_rate: float = 0.0,
    # tie_weights: bool = False,):
    paddle.set_device("cpu")
    model_dict = paddle.load("transformerLM.pdparams")
    tlm.set_state_dict(model_dict)

    tlm.eval()
    #Test the score
    input2 = np.array([5])
    input2 = paddle.to_tensor(input2)
    state = None
    output, state = tlm.score(input2, state, None)

    input3 = np.array([5, 10])
    input3 = paddle.to_tensor(input3)
    output, state = tlm.score(input3, state, None)

    input4 = np.array([5, 10, 0])
    input4 = paddle.to_tensor(input4)
    output, state = tlm.score(input4, state, None)
    print("output", output)
    """
    #Test the batch score
    batch_size = 2
    inp2 = np.array([[5], [10]])
    inp2 = paddle.to_tensor(inp2)
    output, states = tlm.batch_score(
        inp2, [(None,None,0)] * batch_size)
    inp3 = np.array([[100], [30]])
    inp3 = paddle.to_tensor(inp3)
    output, states = tlm.batch_score(
        inp3, states)
    print("output", output)
    #print("cache", cache)
    #np.save("output_pd.npy", output)
    """
