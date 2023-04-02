# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import copy
from collections import defaultdict

import paddle

from paddlespeech.cli.log import logger
from paddlespeech.s2t.utils.utility import log_add

__all__ = ['CTCPrefixBeamSearch']


class CTCPrefixBeamSearch:
    def __init__(self, config):
        """Implement the ctc prefix beam search

        Args:
            config (yacs.config.CfgNode): the ctc prefix beam search configuration
        """
        self.config = config

        # beam size
        self.first_beam_size = self.config.beam_size
        # TODO(support second beam size)
        self.second_beam_size = int(self.first_beam_size * 1.0)
        logger.info(
            f"first and second beam size: {self.first_beam_size}, {self.second_beam_size}"
        )

        # state
        self.cur_hyps = None
        self.hyps = None
        self.abs_time_step = 0

        self.reset()

    def reset(self):
        """Rest the search cache value
        """
        self.cur_hyps = None
        self.hyps = None
        self.abs_time_step = 0

    @paddle.no_grad()
    def search(self, ctc_probs, device, blank_id=0):
        """ctc prefix beam search method decode a chunk feature

        Args:
            xs (paddle.Tensor): feature data
            ctc_probs (paddle.Tensor): the ctc probability of all the tokens
            device (paddle.fluid.core_avx.Place): the feature host device, such as CUDAPlace(0).
            blank_id (int, optional): the blank id in the vocab. Defaults to 0.

        Returns:
            list: the search result
        """
        # decode
        logger.info("start to ctc prefix search")
        assert len(ctc_probs.shape) == 2
        batch_size = 1

        vocab_size = ctc_probs.shape[1]
        first_beam_size = min(self.first_beam_size, vocab_size)
        second_beam_size = min(self.second_beam_size, vocab_size)
        logger.info(
            f"effect first and second beam size: {self.first_beam_size}, {self.second_beam_size}"
        )

        maxlen = ctc_probs.shape[0]

        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        # 0. blank_ending_score,
        # 1. none_blank_ending_score,
        # 2. viterbi_blank ending score,
        # 3. viterbi_non_blank score,
        # 4. current_token_prob,
        # 5. times_viterbi_blank, times_b
        # 6. times_titerbi_non_blank, times_nb
        if self.cur_hyps is None:
            self.cur_hyps = [(tuple(), (0.0, -float('inf'), 0.0, 0.0,
                                        -float('inf'), [], []))]
            # self.cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            next_hyps = defaultdict(
                        lambda: (-float('inf'), -float('inf'), -float('inf'), -float('inf'), -float('inf'), [], []))

            # 2.1 First beam prune: select topk best
            #     do token passing process
            top_k_logp, top_k_index = logp.topk(
                first_beam_size)  # (first_beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb, v_b_s, v_nb_s, cur_token_prob, times_b,
                             times_nb) in self.cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == blank_id:  # blank
                        n_pb, n_pnb, n_v_b, n_v_nb, n_cur_token_prob, n_times_b, n_times_nb = next_hyps[
                            prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])

                        pre_times = times_b if v_b_s > v_nb_s else times_nb
                        n_times_b = copy.deepcopy(pre_times)
                        viterbi_score = v_b_s if v_b_s > v_nb_s else v_nb_s
                        n_v_b = viterbi_score + ps
                        next_hyps[prefix] = (n_pb, n_pnb, n_v_b, n_v_nb,
                                             n_cur_token_prob, n_times_b,
                                             n_times_nb)
                    elif s == last:
                        #  Update *ss -> *s;
                        # case1: *a + a => *a
                        n_pb, n_pnb, n_v_b, n_v_nb, n_cur_token_prob, n_times_b, n_times_nb = next_hyps[
                            prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        if n_v_nb < v_nb_s + ps:
                            n_v_nb = v_nb_s + ps
                            if n_cur_token_prob < ps:
                                n_cur_token_prob = ps
                                n_times_nb = copy.deepcopy(times_nb)
                                n_times_nb[
                                    -1] = self.abs_time_step  # 注意，这里要重新使用绝对时间
                        next_hyps[prefix] = (n_pb, n_pnb, n_v_b, n_v_nb,
                                             n_cur_token_prob, n_times_b,
                                             n_times_nb)

                        # Update *s-s -> *ss, - is for blank
                        # Case 2: *aε + a => *aa
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb, n_v_b, n_v_nb, n_cur_token_prob, n_times_b, n_times_nb = next_hyps[
                            n_prefix]
                        if n_v_nb < v_b_s + ps:
                            n_v_nb = v_b_s + ps
                            n_cur_token_prob = ps
                            n_times_nb = copy.deepcopy(times_b)
                            n_times_nb.append(self.abs_time_step)
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb, n_v_b, n_v_nb,
                                               n_cur_token_prob, n_times_b,
                                               n_times_nb)
                    else:
                        # Case 3: *a + b => *ab, *aε + b => *ab
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb, n_v_b, n_v_nb, n_cur_token_prob, n_times_b, n_times_nb = next_hyps[
                            n_prefix]
                        viterbi_score = v_b_s if v_b_s > v_nb_s else v_nb_s
                        pre_times = times_b if v_b_s > v_nb_s else times_nb
                        if n_v_nb < viterbi_score + ps:
                            n_v_nb = viterbi_score + ps
                            n_cur_token_prob = ps
                            n_times_nb = copy.deepcopy(pre_times)
                            n_times_nb.append(self.abs_time_step)

                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb, n_v_b, n_v_nb,
                                               n_cur_token_prob, n_times_b,
                                               n_times_nb)

            # 2.2 Second beam prune
            next_hyps = sorted(
                next_hyps.items(),
                key=lambda x: log_add([x[1][0], x[1][1]]),
                reverse=True)
            self.cur_hyps = next_hyps[:second_beam_size]

            # 2.3 update the absolute time step
            self.abs_time_step += 1

        self.hyps = [(y[0], log_add([y[1][0], y[1][1]]), y[1][2], y[1][3],
                      y[1][4], y[1][5], y[1][6]) for y in self.cur_hyps]

        logger.info("ctc prefix search success")
        return self.hyps

    def get_one_best_hyps(self):
        """Return the one best result

        Returns:
            list: the one best result, List[str]
        """
        return [self.hyps[0][0]]

    def get_hyps(self):
        """Return the search hyps

        Returns:
            list: return the search hyps, List[Tuple[str, float, ...]]
        """
        return self.hyps

    def finalize_search(self):
        """do nothing in ctc_prefix_beam_search
        """
        pass
