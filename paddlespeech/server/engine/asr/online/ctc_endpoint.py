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
from dataclasses import dataclass

import numpy as np

from paddlespeech.cli.log import logger


@dataclass
class OnlineCTCEndpointRule:
    must_contain_nonsilence: bool = True
    min_trailing_silence: int = 1000
    min_utterance_length: int = 0


@dataclass
class OnlineCTCEndpoingOpt:
    frame_shift_in_ms: int = 10

    blank: int = 0  # blank id, that we consider as silence for purposes of endpointing.
    blank_threshold: float = 0.8  # above blank threshold is silence

    # We support three rules.  We terminate decoding if ANY of these rules
    # evaluates to "true". If you want to add more rules, do it by changing this
    # code.  If you want to disable a rule, you can set the silence-timeout for
    # that rule to a very large number.

    # rule1 times out after 5 seconds of silence, even if we decoded nothing.
    rule1: OnlineCTCEndpointRule = OnlineCTCEndpointRule(False, 5000, 0)
    # rule4 times out after 1.0 seconds of silence after decoding something,
    # even if we did not reach a final-state at all.
    rule2: OnlineCTCEndpointRule = OnlineCTCEndpointRule(True, 1000, 0)
    # rule5 times out after the utterance is 20 seconds long, regardless of
    # anything else.
    rule3: OnlineCTCEndpointRule = OnlineCTCEndpointRule(False, 0, 20000)


class OnlineCTCEndpoint:
    """
    [END-TO-END AUTOMATIC SPEECH RECOGNITION INTEGRATED WITH CTC-BASED VOICE ACTIVITY DETECTION](https://arxiv.org/pdf/2002.00551.pdf)
    """

    def __init__(self, opts: OnlineCTCEndpoingOpt):
        self.opts = opts
        logger.info(f"Endpont Opts: {opts}")
        self.frame_shift_in_ms = opts.frame_shift_in_ms

        self.num_frames_decoded = 0
        self.trailing_silence_frames = 0

        self.reset()

    def reset(self):
        self.num_frames_decoded = 0
        self.trailing_silence_frames = 0

    def rule_activated(self,
                       rule: OnlineCTCEndpointRule,
                       rule_name: str,
                       decoding_something: bool,
                       trailine_silence: int,
                       utterance_length: int) -> bool:
        ans = (
            decoding_something or (not rule.must_contain_nonsilence)
        ) and trailine_silence >= rule.min_trailing_silence and utterance_length >= rule.min_utterance_length
        if (ans):
            logger.info(f"Endpoint Rule: {rule_name} activated: {rule}")
        return ans

    def endpoint_detected(self,
                          ctc_log_probs: np.ndarray,
                          decoding_something: bool) -> bool:
        """detect endpoint.

        Args:
            ctc_log_probs (np.ndarray): (T, D)

        Returns:
            bool: whether endpoint detected.
        """
        for logprob in ctc_log_probs:
            blank_prob = np.exp(logprob[self.opts.blank])

            self.num_frames_decoded += 1
            if blank_prob > self.opts.blank_threshold:
                self.trailing_silence_frames += 1
            else:
                self.trailing_silence_frames = 0

        assert self.num_frames_decoded >= self.trailing_silence_frames
        assert self.frame_shift_in_ms > 0
        
        decoding_something = (self.num_frames_decoded > self.trailing_silence_frames) and decoding_something
        utterance_length = self.num_frames_decoded * self.frame_shift_in_ms
        trailing_silence = self.trailing_silence_frames * self.frame_shift_in_ms

        if self.rule_activated(self.opts.rule1, 'rule1', decoding_something,
                               trailing_silence, utterance_length):
            return True
        if self.rule_activated(self.opts.rule2, 'rule2', decoding_something,
                               trailing_silence, utterance_length):
            return True
        if self.rule_activated(self.opts.rule3, 'rule3', decoding_something,
                               trailing_silence, utterance_length):
            return True
        return False
