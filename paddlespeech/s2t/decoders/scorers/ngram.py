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
"""Ngram lm implement."""
from abc import ABC

import kenlm
import paddle

from .scorer_interface import BatchScorerInterface
from .scorer_interface import PartialScorerInterface


class Ngrambase(ABC):
    """Ngram base implemented through ScorerInterface."""
    def __init__(self, ngram_model, token_list):
        """Initialize Ngrambase.

        Args:
            ngram_model: ngram model path
            token_list: token list from dict or model.json

        """
        self.chardict = [x if x != "<eos>" else "</s>" for x in token_list]
        self.charlen = len(self.chardict)
        self.lm = kenlm.LanguageModel(ngram_model)
        self.tmpkenlmstate = kenlm.State()

    def init_state(self, x):
        """Initialize tmp state."""
        state = kenlm.State()
        self.lm.NullContextWrite(state)
        return state

    def score_partial_(self, y, next_token, state, x):
        """Score interface for both full and partial scorer.

        Args:
            y: previous char
            next_token: next token need to be score
            state: previous state
            x: encoded feature

        Returns:
            tuple[paddle.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        out_state = kenlm.State()
        ys = self.chardict[y[-1]] if y.shape[0] > 1 else "<s>"
        self.lm.BaseScore(state, ys, out_state)
        scores = paddle.empty_like(next_token, dtype=x.dtype)
        for i, j in enumerate(next_token):
            scores[i] = self.lm.BaseScore(out_state, self.chardict[j],
                                          self.tmpkenlmstate)
        return scores, out_state


class NgramFullScorer(Ngrambase, BatchScorerInterface):
    """Fullscorer for ngram."""
    def score(self, y, state, x):
        """Score interface for both full and partial scorer.

        Args:
            y: previous char
            state: previous state
            x: encoded feature

        Returns:
            tuple[paddle.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        return self.score_partial_(y, paddle.to_tensor(range(self.charlen)),
                                   state, x)


class NgramPartScorer(Ngrambase, PartialScorerInterface):
    """Partialscorer for ngram."""
    def score_partial(self, y, next_token, state, x):
        """Score interface for both full and partial scorer.

        Args:
            y: previous char
            next_token: next token need to be score
            state: previous state
            x: encoded feature

        Returns:
            tuple[paddle.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        return self.score_partial_(y, next_token, state, x)

    def select_state(self, state, i):
        """Empty select state for scorer interface."""
        return state
