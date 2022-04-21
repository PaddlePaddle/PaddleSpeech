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
from collections import defaultdict
import paddle
from paddlespeech.cli.log import logger
from paddlespeech.s2t.utils.utility import log_add

__all__ = ['CTCPrefixBeamSearch']


class CTCPrefixBeamSearch:
    def __init__(self, config):
        """Implement the ctc prefix beam search

        Args:
            config (yacs.config.CfgNode): _description_
        """
        self.config = config
        self.reset()

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

        batch_size = 1
        beam_size = self.config.beam_size
        maxlen = ctc_probs.shape[0]

        assert len(ctc_probs.shape) == 2

        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        # blank_ending_score and  none_blank_ending_score in ln domain
        if self.cur_hyps is None:
            self.cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))

            # 2.1 First beam prune: select topk best
            #     do token passing process
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in self.cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == blank_id:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(
                next_hyps.items(),
                key=lambda x: log_add(list(x[1])),
                reverse=True)
            self.cur_hyps = next_hyps[:beam_size]

        self.hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
        logger.info("ctc prefix search success")
        return self.hyps

    def get_one_best_hyps(self):
        """Return the one best result

        Returns:
            list: the one best result
        """
        return [self.hyps[0][0]]

    def get_hyps(self):
        """Return the search hyps

        Returns:
            list: return the search hyps
        """
        return self.hyps

    def reset(self):
        """Rest the search cache value
        """
        self.cur_hyps = None
        self.hyps = None

    def finalize_search(self):
        """do nothing in ctc_prefix_beam_search
        """
        pass
