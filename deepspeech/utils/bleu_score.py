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
"""This module provides functions to calculate bleu score in different level.
e.g. wer for word-level, cer for char-level.
"""
import numpy as np
import sacrebleu

__all__ = ['bleu', 'char_bleu']


def bleu(hypothesis, reference):
    """Calculate BLEU. BLEU compares reference text and
    hypothesis text in word-level using scarebleu.

   

    :param reference: The reference sentences.
    :type reference: list[list[str]]
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: list[str]
    :raises ValueError: If the reference length is zero.
    """

    return sacrebleu.corpus_bleu(hypothesis, reference)

def char_bleu(hypothesis, reference):
    """Calculate BLEU. BLEU compares reference text and
    hypothesis text in char-level using scarebleu.

   

    :param reference: The reference sentences.
    :type reference: list[list[str]]
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: list[str]
    :raises ValueError: If the reference number is zero.
    """
    hypothesis =[' '.join(list(hyp.replace(' ', ''))) for hyp in hypothesis]
    reference = [[' '.join(list(ref_i.replace(' ', ''))) for ref_i in ref ]for ref in reference ]

    return sacrebleu.corpus_bleu(hypothesis, reference)