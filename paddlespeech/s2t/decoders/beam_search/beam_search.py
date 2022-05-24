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
"""Beam search module."""
from itertools import chain
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

import paddle

from ..scorers.scorer_interface import PartialScorerInterface
from ..scorers.scorer_interface import ScorerInterface
from ..utils import end_detect
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: paddle.Tensor  # (T,)
    score: Union[float, paddle.Tensor] = 0
    scores: Dict[str, Union[float, paddle.Tensor]] = dict()
    states: Dict[str, Any] = dict()

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v)
                    for k, v in self.scores.items()}, )._asdict()


class BeamSearch(paddle.nn.Layer):
    """Beam search implementation."""

    def __init__(
            self,
            scorers: Dict[str, ScorerInterface],
            weights: Dict[str, float],
            beam_size: int,
            vocab_size: int,
            sos: int,
            eos: int,
            token_list: List[str]=None,
            pre_beam_ratio: float=1.5,
            pre_beam_score_key: str=None, ):
        """Initialize beam search.

        Args:
            scorers (dict[str, ScorerInterface]): Dict of decoder modules
                e.g., Decoder, CTCPrefixScorer, LM
                The scorer will be ignored if it is `None`
            weights (dict[str, float]): Dict of weights for each scorers
                The scorer will be ignored if its weight is 0
            beam_size (int): The number of hypotheses kept during search
            vocab_size (int): The number of vocabulary
            sos (int): Start of sequence id
            eos (int): End of sequence id
            token_list (list[str]): List of tokens for debug log
            pre_beam_score_key (str): key of scores to perform pre-beam search
            pre_beam_ratio (float): beam size in the pre-beam search
                will be `int(pre_beam_ratio * beam_size)`

        """
        super().__init__()
        # set scorers
        self.weights = weights
        self.scorers = dict()  # all = full + partial
        self.full_scorers = dict()  # full tokens
        self.part_scorers = dict()  # partial tokens
        # this module dict is required for recursive cast
        # `self.to(device, dtype)` in `recog.py`
        self.nn_dict = paddle.nn.LayerDict()  # nn.Layer
        for k, v in scorers.items():
            w = weights.get(k, 0)
            if w == 0 or v is None:
                continue
            assert isinstance(
                v, ScorerInterface
            ), f"{k} ({type(v)}) does not implement ScorerInterface"
            self.scorers[k] = v
            if isinstance(v, PartialScorerInterface):
                self.part_scorers[k] = v
            else:
                self.full_scorers[k] = v
            if isinstance(v, paddle.nn.Layer):
                self.nn_dict[k] = v

        # set configurations
        self.sos = sos
        self.eos = eos
        self.token_list = token_list
        # pre_beam_size > beam_size
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.beam_size = beam_size
        self.n_vocab = vocab_size
        if (pre_beam_score_key is not None and pre_beam_score_key != "full" and
                pre_beam_score_key not in self.full_scorers):
            raise KeyError(
                f"{pre_beam_score_key} is not found in {self.full_scorers}")
        # selected `key` scorer to do pre beam search
        self.pre_beam_score_key = pre_beam_score_key
        # do_pre_beam when need, valid and has part_scorers
        self.do_pre_beam = (self.pre_beam_score_key is not None and
                            self.pre_beam_size < self.n_vocab and
                            len(self.part_scorers) > 0)

    def init_hyp(self, x: paddle.Tensor) -> List[Hypothesis]:
        """Get an initial hypothesis data.

        Args:
            x (paddle.Tensor): The encoder output feature, (T, D)

        Returns:
            Hypothesis: The initial hypothesis.

        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.init_state(x)
            init_scores[k] = 0.0
        return [
            Hypothesis(
                yseq=paddle.to_tensor([self.sos], place=x.place),
                score=0.0,
                scores=init_scores,
                states=init_states, )
        ]

    @staticmethod
    def append_token(xs: paddle.Tensor,
                     x: Union[int, paddle.Tensor]) -> paddle.Tensor:
        """Append new token to prefix tokens.

        Args:
            xs (paddle.Tensor): The prefix token, (T,)
            x (int): The new token to append

        Returns:
            paddle.Tensor: (T+1,), New tensor contains: xs + [x] with xs.dtype and xs.device

        """
        x = paddle.to_tensor([x], dtype=xs.dtype) if isinstance(x, int) else x
        return paddle.concat((xs, x))

    def score_full(self, hyp: Hypothesis, x: paddle.Tensor
                   ) -> Tuple[Dict[str, paddle.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (paddle.Tensor): Corresponding input feature, (T, D)

        Returns:
            Tuple[Dict[str, paddle.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            # scores[k] shape (self.n_vocab,)
            scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x)
        return scores, states

    def score_partial(self,
                      hyp: Hypothesis,
                      ids: paddle.Tensor,
                      x: paddle.Tensor
                      ) -> Tuple[Dict[str, paddle.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.part_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            ids (paddle.Tensor): 1D tensor of new partial tokens to score,
                len(ids) < n_vocab
            x (paddle.Tensor): Corresponding input feature, (T, D)

        Returns:
            Tuple[Dict[str, paddle.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.part_scorers`
                and tensor score values of shape: `(len(ids),)`,
                and state dict that has string keys
                and state values of `self.part_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            # scores[k] shape (len(ids),)
            scores[k], states[k] = d.score_partial(hyp.yseq, ids, hyp.states[k],
                                                   x)
        return scores, states

    def beam(self, weighted_scores: paddle.Tensor,
             ids: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute topk full token ids and partial token ids.

        Args:
            weighted_scores (paddle.Tensor): The weighted sum scores for each tokens.
                Its shape is `(self.n_vocab,)`.
            ids (paddle.Tensor): The partial token ids(Global) to compute topk.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]:
                The topk full token ids and partial token ids.
                Their shapes are `(self.beam_size,)`.
                i.e. (global ids, global relative local ids).

        """
        # no pre beam performed, `ids` equal to `weighted_scores`
        if weighted_scores.shape[0] == ids.shape[0]:
            top_ids = weighted_scores.topk(
                self.beam_size)[1]  # index in n_vocab
            return top_ids, top_ids

        # mask pruned in pre-beam not to select in topk
        tmp = weighted_scores[ids]
        weighted_scores[:] = -float("inf")
        weighted_scores[ids] = tmp
        # top_ids no equal to local_ids, since ids shape not same
        top_ids = weighted_scores.topk(self.beam_size)[1]  # index in n_vocab
        local_ids = weighted_scores[ids].topk(
            self.beam_size)[1]  # index in len(ids)
        return top_ids, local_ids

    @staticmethod
    def merge_scores(
            prev_scores: Dict[str, float],
            next_full_scores: Dict[str, paddle.Tensor],
            full_idx: int,
            next_part_scores: Dict[str, paddle.Tensor],
            part_idx: int, ) -> Dict[str, paddle.Tensor]:
        """Merge scores for new hypothesis.

        Args:
            prev_scores (Dict[str, float]):
                The previous hypothesis scores by `self.scorers`
            next_full_scores (Dict[str, paddle.Tensor]): scores by `self.full_scorers`
            full_idx (int): The next token id for `next_full_scores`
            next_part_scores (Dict[str, paddle.Tensor]):
                scores of partial tokens by `self.part_scorers`
            part_idx (int): The new token id for `next_part_scores`

        Returns:
            Dict[str, paddle.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are scalar tensors by the scorers.

        """
        new_scores = dict()
        for k, v in next_full_scores.items():
            new_scores[k] = prev_scores[k] + v[full_idx]
        for k, v in next_part_scores.items():
            new_scores[k] = prev_scores[k] + v[part_idx]
        return new_scores

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        """Merge states for new hypothesis.

        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`

        Returns:
            Dict[str, paddle.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.

        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, d in self.part_scorers.items():
            new_states[k] = d.select_state(part_states[k], part_idx)
        return new_states

    def search(self, running_hyps: List[Hypothesis],
               x: paddle.Tensor) -> List[Hypothesis]:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (paddle.Tensor): Encoded speech feature (T, D)

        Returns:
            List[Hypotheses]: Best sorted hypotheses

        """
        best_hyps = []
        part_ids = paddle.arange(self.n_vocab)  # no pre-beam
        for hyp in running_hyps:
            # scoring
            weighted_scores = paddle.zeros([self.n_vocab], dtype=x.dtype)
            scores, states = self.score_full(hyp, x)
            for k in self.full_scorers:
                weighted_scores += self.weights[k] * scores[k]
            # partial scoring
            if self.do_pre_beam:
                pre_beam_scores = (weighted_scores
                                   if self.pre_beam_score_key == "full" else
                                   scores[self.pre_beam_score_key])
                part_ids = paddle.topk(pre_beam_scores, self.pre_beam_size)[1]
            part_scores, part_states = self.score_partial(hyp, part_ids, x)
            for k in self.part_scorers:
                weighted_scores[part_ids] += self.weights[k] * part_scores[k]
            # add previous hyp score
            weighted_scores += hyp.score

            # update hyps
            for j, part_j in zip(*self.beam(weighted_scores, part_ids)):
                # `part_j` is `j` relative id in `part_scores`
                # will be (2 x beam at most)
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(hyp.scores, scores, j,
                                                 part_scores, part_j),
                        states=self.merge_states(states, part_states, part_j),
                    ))

            # sort and prune 2 x beam -> beam
            best_hyps = sorted(
                best_hyps, key=lambda x: x.score,
                reverse=True)[:min(len(best_hyps), self.beam_size)]
        return best_hyps

    def forward(self,
                x: paddle.Tensor,
                maxlenratio: float=0.0,
                minlenratio: float=0.0) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (paddle.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                    to automatically find maximum hypothesis lengths
                If maxlenratio<0.0, its absolute value is interpreted
                    as a constant max output length.
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * x.shape[0]))
        minlen = int(minlenratio * x.shape[0])
        logger.info("decoder input length: " + str(x.shape[0]))
        logger.info("max output length: " + str(maxlen))
        logger.info("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(x)
        ended_hyps = []
        for i in range(maxlen):
            logger.debug("position " + str(i))
            best = self.search(running_hyps, x)
            # post process of one iteration
            running_hyps = self.post_process(i, maxlen, maxlenratio, best,
                                             ended_hyps)
            # end detection
            if maxlenratio == 0.0 and end_detect(
                [h.asdict() for h in ended_hyps], i):
                logger.info(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logger.info("no hypothesis. Finish decoding.")
                break
            else:
                logger.debug(f"remained hypotheses: {len(running_hyps)}")

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logger.warning("there is no N-best results, perform recognition "
                           "again with smaller minlenratio.")
            return ([] if minlenratio < 0.1 else
                    self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1)))

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logger.info(
                f"{float(v):6.2f} * {self.weights[k]:3} = {float(v) * self.weights[k]:6.2f} for {k}"
            )
        logger.info(f"total log probability: {float(best.score):.2f}")
        logger.info(
            f"normalized log probability: {float(best.score) / len(best.yseq):.2f}"
        )
        logger.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            # logger.info(
            #     "best hypo: "
            #     + "".join([self.token_list[x] for x in best.yseq[1:-1]])
            #     + "\n"
            # )
            logger.info("best hypo: " + "".join(
                [self.token_list[x] for x in best.yseq[1:]]) + "\n")
        return nbest_hyps

    def post_process(
            self,
            i: int,
            maxlen: int,
            maxlenratio: float,
            running_hyps: List[Hypothesis],
            ended_hyps: List[Hypothesis], ) -> List[Hypothesis]:
        """Perform post-processing of beam search iterations.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (List[Hypothesis]): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.

        Returns:
            List[Hypothesis]: The new running hypotheses.

        """
        logger.debug(f"the number of running hypotheses: {len(running_hyps)}")
        if self.token_list is not None:
            logger.debug("best hypo: " + "".join(
                [self.token_list[x] for x in running_hyps[0].yseq[1:]]))
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logger.info("adding <eos> in the last position in the loop")
            running_hyps = [
                h._replace(yseq=self.append_token(h.yseq, self.eos))
                for h in running_hyps
            ]

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a problem, number of hyps < beam)
        remained_hyps = []
        for hyp in running_hyps:
            if hyp.yseq[-1] == self.eos:
                # e.g., Word LM needs to add final <eos> score
                for k, d in chain(self.full_scorers.items(),
                                  self.part_scorers.items()):
                    s = d.final_score(hyp.states[k])
                    hyp.scores[k] += s
                    hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
                ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)
        return remained_hyps


def beam_search(
        x: paddle.Tensor,
        sos: int,
        eos: int,
        beam_size: int,
        vocab_size: int,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        token_list: List[str]=None,
        maxlenratio: float=0.0,
        minlenratio: float=0.0,
        pre_beam_ratio: float=1.5,
        pre_beam_score_key: str="full", ) -> list:
    """Perform beam search with scorers.

    Args:
        x (paddle.Tensor): Encoded speech feature (T, D)
        sos (int): Start of sequence id
        eos (int): End of sequence id
        beam_size (int): The number of hypotheses kept during search
        vocab_size (int): The number of vocabulary
        scorers (dict[str, ScorerInterface]): Dict of decoder modules
            e.g., Decoder, CTCPrefixScorer, LM
            The scorer will be ignored if it is `None`
        weights (dict[str, float]): Dict of weights for each scorers
            The scorer will be ignored if its weight is 0
        token_list (list[str]): List of tokens for debug log
        maxlenratio (float): Input length ratio to obtain max output length.
            If maxlenratio=0.0 (default), it uses a end-detect function
            to automatically find maximum hypothesis lengths
        minlenratio (float): Input length ratio to obtain min output length.
        pre_beam_score_key (str): key of scores to perform pre-beam search
        pre_beam_ratio (float): beam size in the pre-beam search
            will be `int(pre_beam_ratio * beam_size)`

    Returns:
        List[Dict]: N-best decoding results

    """
    ret = BeamSearch(
        scorers,
        weights,
        beam_size=beam_size,
        vocab_size=vocab_size,
        pre_beam_ratio=pre_beam_ratio,
        pre_beam_score_key=pre_beam_score_key,
        sos=sos,
        eos=eos,
        token_list=token_list, ).forward(
            x=x, maxlenratio=maxlenratio, minlenratio=minlenratio)
    return [h.asdict() for h in ret]
