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

import math
import collections
import numpy as np
import logging
from typing import Optional
from yacs.config import CfgNode

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

from deepspeech.modules.conv import ConvStack
from deepspeech.modules.rnn import RNNStack
from deepspeech.modules.mask import sequence_mask
from deepspeech.modules.activation import brelu
from deepspeech.utils import checkpoint
from deepspeech.decoders.swig_wrapper import Scorer
from deepspeech.decoders.swig_wrapper import ctc_greedy_decoder
from deepspeech.decoders.swig_wrapper import ctc_beam_search_decoder_batch

logger = logging.getLogger(__name__)

__all__ = ['DeepSpeech2Model']


class DeepSpeech2Model(nn.Layer):
    """The DeepSpeech2 network structure.

    :param audio_data: Audio spectrogram data layer.
    :type audio_data: Variable
    :param text_data: Transcription text data layer.
    :type text_data: Variable
    :param audio_len: Valid sequence length data layer.
    :type audio_len: Variable
    :param masks: Masks data layer to reset padding.
    :type masks: Variable
    :param dict_size: Dictionary size for tokenized transcription.
    :type dict_size: int
    :param num_conv_layers: Number of stacking convolution layers.
    :type num_conv_layers: int
    :param num_rnn_layers: Number of stacking RNN layers.
    :type num_rnn_layers: int
    :param rnn_size: RNN layer size (dimension of RNN cells).
    :type rnn_size: int
    :param use_gru: Use gru if set True. Use simple rnn if set False.
    :type use_gru: bool
    :param share_rnn_weights: Whether to share input-hidden weights between
                              forward and backward direction RNNs.
                              It is only available when use_gru=False.
    :type share_weights: bool
    :return: A tuple of an output unnormalized log probability layer (
             before softmax) and a ctc cost layer.
    :rtype: tuple of LayerOutput    
    """

    @classmethod
    def params(cls, config: Optional[CfgNode]=None) -> CfgNode:
        default = CfgNode(
            dict(
                num_conv_layers=2,  #Number of stacking convolution layers.
                num_rnn_layers=3,  #Number of stacking RNN layers.
                rnn_layer_size=1024,  #RNN layer size (number of RNN cells).
                use_gru=True,  #Use gru if set True. Use simple rnn if set False.
                share_rnn_weights=True  #Whether to share input-hidden weights between forward and backward directional RNNs.Notice that for GRU, weight sharing is not supported.
            ))
        if config is not None:
            config.merge_from_other_cfg(default)
        return default

    def __init__(self,
                 feat_size,
                 dict_size,
                 num_conv_layers=2,
                 num_rnn_layers=3,
                 rnn_size=1024,
                 use_gru=False,
                 share_rnn_weights=True):
        super().__init__()
        self.feat_size = feat_size  # 161 for linear
        self.dict_size = dict_size

        self.conv = ConvStack(feat_size, num_conv_layers)

        i_size = self.conv.output_height  # H after conv stack
        self.rnn = RNNStack(
            i_size=i_size,
            h_size=rnn_size,
            num_stacks=num_rnn_layers,
            use_gru=use_gru,
            share_rnn_weights=share_rnn_weights)
        self.fc = nn.Linear(rnn_size * 2, dict_size + 1)

        self.logger = logging.getLogger(__name__)
        self._ext_scorer = None

    def infer(self, audio, audio_len):
        # [B, D, T] -> [B, C=1, D, T]
        audio = audio.unsqueeze(1)

        # convolution group
        x, audio_len = self.conv(audio, audio_len)
        #print('conv out', x.shape)

        # convert data from convolution feature map to sequence of vectors
        B, C, D, T = paddle.shape(x)
        x = x.transpose([0, 3, 1, 2])  #[B, T, C, D]
        x = x.reshape([B, T, C * D])  #[B, T, C*D]
        #print('rnn input', x.shape)

        # remove padding part
        x, audio_len = self.rnn(x, audio_len)  #[B, T, D]
        #print('rnn output', x.shape)

        logits = self.fc(x)  #[B, T, V + 1]

        #ctcdecoder need probs, not log_probs
        probs = F.softmax(logits)

        return logits, probs, audio_len

    def forward(self, audio, text, audio_len, text_len):
        """
        audio: shape [B, D, T]
        text: shape [B, T]
        audio_len: shape [B]
        text_len: shape [B]
        """
        return self.infer(audio, audio_len)

    @paddle.no_grad()
    def predict(self, audio, audio_len):
        """ Model infer """
        return self.infer(audio, audio_len)

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
            self.logger.info("begin to initialize the external scorer "
                             "for decoding")
            self._ext_scorer = Scorer(beam_alpha, beam_beta,
                                      language_model_path, vocab_list)
            lm_char_based = self._ext_scorer.is_character_based()
            lm_max_order = self._ext_scorer.get_max_order()
            lm_dict_size = self._ext_scorer.get_dict_size()
            self.logger.info("language model: "
                             "is_character_based = %d," % lm_char_based +
                             " max_order = %d," % lm_max_order +
                             " dict_size = %d" % lm_dict_size)
            self.logger.info("end initializing scorer")
        else:
            self._ext_scorer = None
            self.logger.info("no language model provided, "
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

    @paddle.no_grad()
    def decode(self, audio, audio_len, vocab_list, decoding_method,
               lang_model_path, beam_alpha, beam_beta, beam_size, cutoff_prob,
               cutoff_top_n, num_processes):
        _, probs, logits_lens = self.predict(audio, audio_len)
        return self.decode_probs(probs.numpy(), logits_lens, vocab_list,
                                 decoding_method, lang_model_path, beam_alpha,
                                 beam_beta, beam_size, cutoff_prob,
                                 cutoff_top_n, num_processes)

    def from_pretrained(self, checkpoint_path):
        """Build a model from a pretrained model.
        Parameters
        ----------
        model: nn.Layer
            Asr Model.
        
        checkpoint_path: Path or str
            The path of pretrained model checkpoint, without extension name.
        
        Returns
        -------
        Model
            The model build from pretrined result.
        """
        checkpoint.load_parameters(self, checkpoint_path=checkpoint_path)
        return
