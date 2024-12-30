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
"""Scorer interface module."""
import warnings
from typing import Any
from typing import List
from typing import Tuple

import paddle


class ScorerInterface:
    """Scorer interface for beam search.

    The scorer performs scoring of the all tokens in vocabulary.

    Examples:
        * Search heuristics
            * :class:`scorers.length_bonus.LengthBonus`
        * Decoder networks of the sequence-to-sequence models
            * :class:`transformer.decoder.Decoder`
            * :class:`rnn.decoders.Decoder`
        * Neural language models
            * :class:`lm.transformer.TransformerLM`
            * :class:`lm.default.DefaultRNNLM`
            * :class:`lm.seq_rnn.SequentialRNNLM`

    """

    def init_state(self, x: paddle.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (paddle.Tensor): The encoded feature tensor

        Returns: initial state

        """
        return None

    def select_state(self, state: Any, i: int, new_id: int=None) -> Any:
        """Select state with relative ids in the main beam search.

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label index to select a state if necessary

        Returns:
            state: pruned state

        """
        return None if state is None else state[i]

    def score(self, y: paddle.Tensor, state: Any,
              x: paddle.Tensor) -> Tuple[paddle.Tensor, Any]:
        """Score new token (required).

        Args:
            y (paddle.Tensor): 1D paddle.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (paddle.Tensor): The encoder feature that generates ys.

        Returns:
            tuple[paddle.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys

        """
        raise NotImplementedError

    def final_score(self, state: Any) -> float:
        """Score eos (optional).

        Args:
            state: Scorer state for prefix tokens

        Returns:
            float: final score

        """
        return 0.0


class BatchScorerInterface(ScorerInterface):
    """Batch scorer interface."""

    def batch_init_state(self, x: paddle.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (paddle.Tensor): The encoded feature tensor

        Returns: initial state

        """
        return self.init_state(x)

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
        warnings.warn(
            "{} batch score is implemented through for loop not parallelized".
            format(self.__class__.__name__))
        scores = list()
        outstates = list()
        for i, (y, state, x) in enumerate(zip(ys, states, xs)):
            score, outstate = self.score(y, state, x)
            outstates.append(outstate)
            scores.append(score)
        scores = paddle.cat(scores, 0).reshape([ys.shape[0], -1])
        return scores, outstates


class PartialScorerInterface(ScorerInterface):
    """Partial scorer interface for beam search.

    The partial scorer performs scoring when non-partial scorer finished scoring,
    and receives pre-pruned next tokens to score because it is too heavy to score
    all the tokens.

    Score sub-set of tokens, not all.

    Examples:
         * Prefix search for connectionist-temporal-classification models
             * :class:`decoders.scorers.ctc.CTCPrefixScorer`

    """

    def score_partial(self,
                      y: paddle.Tensor,
                      next_tokens: paddle.Tensor,
                      state: Any,
                      x: paddle.Tensor) -> Tuple[paddle.Tensor, Any]:
        """Score new token (required).

        Args:
            y (paddle.Tensor): 1D prefix token
            next_tokens (paddle.Tensor): paddle.int64 next token to score
            state: decoder state for prefix tokens
            x (paddle.Tensor): The encoder feature that generates ys

        Returns:
            tuple[paddle.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        raise NotImplementedError


class BatchPartialScorerInterface(BatchScorerInterface, PartialScorerInterface):
    """Batch partial scorer interface for beam search."""

    def batch_score_partial(
            self,
            ys: paddle.Tensor,
            next_tokens: paddle.Tensor,
            states: List[Any],
            xs: paddle.Tensor, ) -> Tuple[paddle.Tensor, Any]:
        """Score new token (required).

        Args:
            ys (paddle.Tensor): paddle.int64 prefix tokens (n_batch, ylen).
            next_tokens (paddle.Tensor): paddle.int64 tokens to score (n_batch, n_token).
            states (List[Any]): Scorer states for prefix tokens.
            xs (paddle.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[paddle.Tensor, Any]:
                Tuple of a score tensor for ys that has a shape `(n_batch, n_vocab)`
                and next states for ys
        """
        raise NotImplementedError
