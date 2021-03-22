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

import logging
from typeguard import check_argument_types

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

from deepspeech.decoders.swig_wrapper import Scorer
from deepspeech.decoders.swig_wrapper import ctc_greedy_decoder
from deepspeech.decoders.swig_wrapper import ctc_beam_search_decoder_batch
from deepspeech.modules.loss import CTCLoss

logger = logging.getLogger(__name__)

__all__ = ['CTCDecoder']


class CTCDecoder(nn.Layer):
    def __init__(self,
                 enc_n_units,
                 odim,
                 blank_id=0,
                 dropout_rate: float=0.0,
                 reduction: bool=True,
                 batch_average: bool=False):
        """CTC decoder

        Args:
            enc_n_units ([int]): encoder output dimention
            vocab_size ([int]): text vocabulary size
            dropout_rate (float): dropout rate (0.0 ~ 1.0)
            reduction (bool): reduce the CTC loss into a scalar, True for 'sum' or 'none'
            batch_average (bool): do batch dim wise average.
        """
        assert check_argument_types()
        super().__init__()

        self.blank_id = blank_id
        self.odim = odim
        self.dropout_rate = dropout_rate
        self.ctc_lo = nn.Linear(enc_n_units, self.odim)
        reduction_type = "sum" if reduction else "none"
        self.criterion = CTCLoss(
            blank=self.blank_id,
            reduction=reduction_type,
            batch_average=batch_average)

        # CTCDecoder LM Score handle
        self._ext_scorer = None

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate CTC loss.

        Args:
            hs_pad (Tensor): batch of padded hidden state sequences (B, Tmax, D)
            hlens (Tensor): batch of lengths of hidden state sequences (B)
            ys_pad (Tenosr): batch of padded character id sequence tensor (B, Lmax)
            ys_lens (Tensor): batch of lengths of character sequence (B)
        Returns:
            loss (Tenosr): scalar.
        """
        logits = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        loss = self.criterion(logits, ys_pad, hlens, ys_lens)
        return loss

    def probs(self, eouts: paddle.Tensor, temperature: float=1.0):
        """Get CTC probabilities.
        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
        Returns:
            probs (FloatTensor): `[B, T, odim]`
        """
        return F.softmax(self.ctc_lo(eouts) / temperature, axis=-1)

    def scores(self, eouts: paddle.Tensor, temperature: float=1.0):
        """Get log-scale CTC probabilities.
        Args:
            eouts (FloatTensor): `[B, T, enc_units]`
        Returns:
            log_probs (FloatTensor): `[B, T, odim]`
        """
        return F.log_softmax(self.ctc_lo(eouts) / temperature, axis=-1)

    def log_softmax(self, hs_pad: paddle.Tensor) -> paddle.Tensor:
        """log_softmax of frame activations
        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            paddle.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return self.scores(hs_pad)

    def argmax(self, hs_pad: paddle.Tensor) -> paddle.Tensor:
        """argmax of frame activations
        Args:
            paddle.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            paddle.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return paddle.argmax(self.ctc_lo(hs_pad), dim=2)

    def _decode_batch_greedy(self, probs_split, vocab_list):
        """Decode by best path for a batch of probs matrix input.
        :param probs_split: List of 2-D probability matrix, and each consists
                            of prob vectors for one speech utterancce.
        :param probs_split: List of matrix
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        :return: List of transcription texts.
        :rtype: List of str
        """
        results = []
        for i, probs in enumerate(probs_split):
            output_transcription = ctc_greedy_decoder(
                probs_seq=probs, vocabulary=vocab_list)
            results.append(output_transcription)
        return results

    def _init_ext_scorer(self, beam_alpha, beam_beta, language_model_path,
                         vocab_list):
        """Initialize the external scorer.
        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param language_model_path: Filepath for language model. If it is
                                    empty, the external scorer will be set to
                                    None, and the decoding method will be pure
                                    beam search without scorer.
        :type language_model_path: str|None
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        """
        # init once
        if self._ext_scorer != None:
            return

        if language_model_path != '':
            logger.info("begin to initialize the external scorer "
                        "for decoding")
            self._ext_scorer = Scorer(beam_alpha, beam_beta,
                                      language_model_path, vocab_list)
            lm_char_based = self._ext_scorer.is_character_based()
            lm_max_order = self._ext_scorer.get_max_order()
            lm_dict_size = self._ext_scorer.get_dict_size()
            logger.info("language model: "
                        "is_character_based = %d," % lm_char_based +
                        " max_order = %d," % lm_max_order + " dict_size = %d" %
                        lm_dict_size)
            logger.info("end initializing scorer")
        else:
            self._ext_scorer = None
            logger.info("no language model provided, "
                        "decoding by pure beam search without scorer.")

    def _decode_batch_beam_search(self, probs_split, beam_alpha, beam_beta,
                                  beam_size, cutoff_prob, cutoff_top_n,
                                  vocab_list, num_processes):
        """Decode by beam search for a batch of probs matrix input.
        :param probs_split: List of 2-D probability matrix, and each consists
                            of prob vectors for one speech utterancce.
        :param probs_split: List of matrix
        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param beam_size: Width for Beam search.
        :type beam_size: int
        :param cutoff_prob: Cutoff probability in pruning,
                            default 1.0, no pruning.
        :type cutoff_prob: float
        :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                        characters with highest probs in vocabulary will be
                        used in beam search, default 40.
        :type cutoff_top_n: int
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        :param num_processes: Number of processes (CPU) for decoder.
        :type num_processes: int
        :return: List of transcription texts.
        :rtype: List of str
        """
        if self._ext_scorer != None:
            self._ext_scorer.reset_params(beam_alpha, beam_beta)

        # beam search decode
        num_processes = min(num_processes, len(probs_split))
        beam_search_results = ctc_beam_search_decoder_batch(
            probs_split=probs_split,
            vocabulary=vocab_list,
            beam_size=beam_size,
            num_processes=num_processes,
            ext_scoring_func=self._ext_scorer,
            cutoff_prob=cutoff_prob,
            cutoff_top_n=cutoff_top_n)

        results = [result[0][1] for result in beam_search_results]
        return results

    def init_decode(self, beam_alpha, beam_beta, lang_model_path, vocab_list,
                    decoding_method):
        if decoding_method == "ctc_beam_search":
            self._init_ext_scorer(beam_alpha, beam_beta, lang_model_path,
                                  vocab_list)

    def decode_probs(self, probs, logits_lens, vocab_list, decoding_method,
                     lang_model_path, beam_alpha, beam_beta, beam_size,
                     cutoff_prob, cutoff_top_n, num_processes):
        """ probs: activation after softmax 
        logits_len: audio output lens
        """
        probs_split = [probs[i, :l, :] for i, l in enumerate(logits_lens)]
        if decoding_method == "ctc_greedy":
            result_transcripts = self._decode_batch_greedy(
                probs_split=probs_split, vocab_list=vocab_list)
        elif decoding_method == "ctc_beam_search":
            result_transcripts = self._decode_batch_beam_search(
                probs_split=probs_split,
                beam_alpha=beam_alpha,
                beam_beta=beam_beta,
                beam_size=beam_size,
                cutoff_prob=cutoff_prob,
                cutoff_top_n=cutoff_top_n,
                vocab_list=vocab_list,
                num_processes=num_processes)
        else:
            raise ValueError(f"Not support: {decoding_method}")
        return result_transcripts
