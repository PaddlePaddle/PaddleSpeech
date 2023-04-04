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
"""ScorerInterface implementation for CTC."""
import numpy as np
import paddle

from .ctc_prefix_score import CTCPrefixScore
from .ctc_prefix_score import CTCPrefixScorePD
from .scorer_interface import BatchPartialScorerInterface


class CTCPrefixScorer(BatchPartialScorerInterface):
    """Decoder interface wrapper for CTCPrefixScore."""
    def __init__(self, ctc: paddle.nn.Layer, eos: int):
        """Initialize class.

        Args:
            ctc (paddle.nn.Layer): The CTC implementation.
                For example, :class:`paddlespeech.s2t.modules.ctc.CTC`
            eos (int): The end-of-sequence id.

        """
        self.ctc = ctc
        self.eos = eos
        self.impl = None

    def init_state(self, x: paddle.Tensor):
        """Get an initial state for decoding.

        Args:
            x (paddle.Tensor): The encoded feature tensor

        Returns: initial state

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0)).squeeze(0).numpy()
        # TODO(karita): use CTCPrefixScorePD
        self.impl = CTCPrefixScore(logp, 0, self.eos, np)
        return 0, self.impl.initial_state()

    def select_state(self, state, i, new_id=None):
        """Select state with relative ids in the main beam search.

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label id to select a state if necessary

        Returns:
            state: pruned state

        """
        if type(state) == tuple:
            if len(state) == 2:  # for CTCPrefixScore
                sc, st = state
                return sc[i], st[i]
            else:  # for CTCPrefixScorePD (need new_id > 0)
                r, log_psi, f_min, f_max, scoring_idmap = state
                s = log_psi[i, new_id].expand(paddle.shape(log_psi)[1])
                if scoring_idmap is not None:
                    return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
                else:
                    return r[:, :, i, new_id], s, f_min, f_max
        return None if state is None else state[i]

    def score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (paddle.Tensor): 1D prefix token
            next_tokens (paddle.Tensor): paddle.int64 next token to score
            state: decoder state for prefix tokens
            x (paddle.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[paddle.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        prev_score, state = state
        presub_score, new_st = self.impl(y.cpu(), ids.cpu(), state)
        tscore = paddle.to_tensor(presub_score - prev_score,
                                  place=x.place,
                                  dtype=x.dtype)
        return tscore, (presub_score, new_st)

    def batch_init_state(self, x: paddle.Tensor):
        """Get an initial state for decoding.

        Args:
            x (paddle.Tensor): The encoded feature tensor

        Returns: initial state

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0))  # assuming batch_size = 1
        xlen = paddle.to_tensor([paddle.shape(logp)[1]])
        self.impl = CTCPrefixScorePD(logp, xlen, 0, self.eos)
        return None

    def batch_score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (paddle.Tensor): 1D prefix token
            ids (paddle.Tensor): paddle.int64 next token to score
            state: decoder state for prefix tokens
            x (paddle.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[paddle.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        batch_state = ((
            paddle.stack([s[0] for s in state], axis=2),
            paddle.stack([s[1] for s in state]),
            state[0][2],
            state[0][3],
        ) if state[0] is not None else None)
        return self.impl(y, batch_state, ids)

    def extend_prob(self, x: paddle.Tensor):
        """Extend probs for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            x (paddle.Tensor): The encoded feature tensor

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0))
        self.impl.extend_prob(logp)

    def extend_state(self, state):
        """Extend state for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            state: The states of hyps

        Returns: extended state

        """
        new_state = []
        for s in state:
            new_state.append(self.impl.extend_state(s))

        return new_state
