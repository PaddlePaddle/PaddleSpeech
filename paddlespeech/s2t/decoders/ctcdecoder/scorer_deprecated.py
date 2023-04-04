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
"""External Scorer for Beam Search Decoder."""
import os

import kenlm
import numpy as np


class Scorer(object):
    """External scorer to evaluate a prefix or whole sentence in
       beam search decoding, including the score from n-gram language
       model and word count.

    :param alpha: Parameter associated with language model. Don't use
                  language model when alpha = 0.
    :type alpha: float
    :param beta: Parameter associated with word count. Don't use word
                count when beta = 0.
    :type beta: float
    :model_path: Path to load language model.
    :type model_path: str
    """
    def __init__(self, alpha, beta, model_path):
        self._alpha = alpha
        self._beta = beta
        if not os.path.isfile(model_path):
            raise IOError("Invaid language model path: %s" % model_path)
        self._language_model = kenlm.LanguageModel(model_path)

    # n-gram language model scoring
    def _language_model_score(self, sentence):
        #log10 prob of last word
        log_cond_prob = list(
            self._language_model.full_scores(sentence, eos=False))[-1][0]
        return np.power(10, log_cond_prob)

    # word insertion term
    def _word_count(self, sentence):
        words = sentence.strip().split(' ')
        return len(words)

    # reset alpha and beta
    def reset_params(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    # execute evaluation
    def __call__(self, sentence, log=False):
        """Evaluation function, gathering all the different scores
        and return the final one.

        :param sentence: The input sentence for evaluation
        :type sentence: str
        :param log: Whether return the score in log representation.
        :type log: bool
        :return: Evaluation score, in the decimal or log.
        :rtype: float
        """
        lm = self._language_model_score(sentence)
        word_cnt = self._word_count(sentence)
        if log is False:
            score = np.power(lm, self._alpha) * np.power(word_cnt, self._beta)
        else:
            score = self._alpha * np.log(lm) + self._beta * np.log(word_cnt)
        return score
