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
"""This module provides functions to calculate bleu score in different level.
e.g. wer for word-level, cer for char-level.
"""
import numpy as np
import sacrebleu

__all__ = ['bleu', 'char_bleu', "ErrorCalculator"]


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
    hypothesis = [' '.join(list(hyp.replace(' ', ''))) for hyp in hypothesis]
    reference = [[' '.join(list(ref_i.replace(' ', ''))) for ref_i in ref]
                 for ref in reference]

    return sacrebleu.corpus_bleu(hypothesis, reference)


class ErrorCalculator():
    """Calculate BLEU for ST and MT models during training.

    :param y_hats: numpy array with predicted text
    :param y_pads: numpy array with true (target) text
    :param char_list: vocabulary list
    :param sym_space: space symbol
    :param sym_pad: pad symbol
    :param report_bleu: report BLUE score if True
    """
    def __init__(self, char_list, sym_space, sym_pad, report_bleu=False):
        """Construct an ErrorCalculator object."""
        super().__init__()
        self.char_list = char_list
        self.space = sym_space
        self.pad = sym_pad
        self.report_bleu = report_bleu
        if self.space in self.char_list:
            self.idx_space = self.char_list.index(self.space)
        else:
            self.idx_space = None

    def __call__(self, ys_hat, ys_pad):
        """Calculate corpus-level BLEU score.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :return: corpus-level BLEU score in a mini-batch
        :rtype float
        """
        bleu = None
        if not self.report_bleu:
            return bleu

        bleu = self.calculate_corpus_bleu(ys_hat, ys_pad)
        return bleu

    def calculate_corpus_bleu(self, ys_hat, ys_pad):
        """Calculate corpus-level BLEU score in a mini-batch.

        :param torch.Tensor seqs_hat: prediction (batch, seqlen)
        :param torch.Tensor seqs_true: reference (batch, seqlen)
        :return: corpus-level BLEU score
        :rtype float
        """
        seqs_hat, seqs_true = [], []
        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]
            eos_true = np.where(y_true == -1)[0]
            ymax = eos_true[0] if len(eos_true) > 0 else len(y_true)
            # NOTE: padding index (-1) in y_true is used to pad y_hat
            # because y_hats is not padded with -1
            seq_hat = [self.char_list[int(idx)] for idx in y_hat[:ymax]]
            seq_true = [
                self.char_list[int(idx)] for idx in y_true if int(idx) != -1
            ]
            seq_hat_text = "".join(seq_hat).replace(self.space, " ")
            seq_hat_text = seq_hat_text.replace(self.pad, "")
            seq_true_text = "".join(seq_true).replace(self.space, " ")
            seqs_hat.append(seq_hat_text)
            seqs_true.append(seq_true_text)
        bleu = sacrebleu.corpus_bleu(seqs_hat, [[ref] for ref in seqs_true])
        return bleu.score * 100
