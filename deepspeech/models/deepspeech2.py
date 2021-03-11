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

from deepspeech.modules.mask import sequence_mask
from deepspeech.modules.activation import brelu
from deepspeech.modules.conv import ConvStack
from deepspeech.modules.rnn import RNNStack
from deepspeech.modules.ctc import CTCDecoder

from deepspeech.utils import checkpoint
from deepspeech.utils import layer_tools

logger = logging.getLogger(__name__)

__all__ = ['DeepSpeech2Model']


class CRNNEncoder(nn.Layer):
    def __init__(self,
                 feat_size,
                 dict_size,
                 num_conv_layers=2,
                 num_rnn_layers=3,
                 rnn_size=1024,
                 use_gru=False,
                 share_rnn_weights=True):
        super().__init__()
        self.rnn_size = rnn_size
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

    @property
    def output_size(self):
        return self.rnn_size * 2

    def forward(self, audio, audio_len):
        """
        audio: shape [B, D, T]
        text: shape [B, T]
        audio_len: shape [B]
        text_len: shape [B]
        """
        """Compute Encoder outputs

        Args:
            audio (Tensor): [B, D, T]
            text (Tensor): [B, T]
            audio_len (Tensor): [B]
            text_len (Tensor): [B]
        Returns:
            x (Tensor): encoder outputs, [B, T, D]
            x_lens (Tensor): encoder length, [B]
        """
        # [B, D, T] -> [B, C=1, D, T]
        x = audio.unsqueeze(1)
        x_lens = audio_len

        # convolution group
        x, x_lens = self.conv(x, x_lens)

        # convert data from convolution feature map to sequence of vectors
        #B, C, D, T = paddle.shape(x)  # not work under jit
        x = x.transpose([0, 3, 1, 2])  #[B, T, C, D]
        #x = x.reshape([B, T, C * D])  #[B, T, C*D]  # not work under jit
        x = x.reshape([0, 0, -1])  #[B, T, C*D]

        # remove padding part
        x, x_lens = self.rnn(x, x_lens)  #[B, T, D]
        return x, x_lens


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
        self.encoder = CRNNEncoder(
            feat_size=feat_size,
            dict_size=dict_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            rnn_size=rnn_size,
            use_gru=use_gru,
            share_rnn_weights=share_rnn_weights)
        assert (self.encoder.output_size == rnn_size * 2)

        self.decoder = CTCDecoder(
            enc_n_units=self.encoder.output_size,
            odim=dict_size + 1,  # <blank> is append after vocab
            blank_id=dict_size,  # last token is <blank>
            dropout_rate=0.0,
            reduction=True)

    def forward(self, audio, text, audio_len, text_len):
        """Compute Model loss

        Args:
            audio (Tenosr): [B, D, T]
            text (Tensor): [B, T]
            audio_len (Tensor): [B]
            text_len (Tensor): [B]

        Returns:
            loss (Tenosr): [1]
        """

        eouts, eouts_len = self.encoder(audio, audio_len)
        loss = self.decoder(eouts, eouts_len, text, text_len)
        return loss

    @paddle.no_grad()
    def decode(self, audio, audio_len, vocab_list, decoding_method,
               lang_model_path, beam_alpha, beam_beta, beam_size, cutoff_prob,
               cutoff_top_n, num_processes):
        # init once
        # decoders only accept string encoded in utf-8
        self.decoder.init_decode(
            beam_alpha=beam_alpha,
            beam_beta=beam_beta,
            lang_model_path=lang_model_path,
            vocab_list=vocab_list,
            decoding_method=decoding_method)

        eouts, eouts_len = self.encoder(audio, audio_len)
        probs = self.decoder.probs(eouts)
        return self.decoder.decode_probs(
            probs.numpy(), eouts_len, vocab_list, decoding_method,
            lang_model_path, beam_alpha, beam_beta, beam_size, cutoff_prob,
            cutoff_top_n, num_processes)

    @classmethod
    def from_pretrained(cls, dataset, config, checkpoint_path):
        """Build a DeepSpeech2Model model from a pretrained model.
        Parameters
        ----------
        dataset: paddle.io.Dataset

        config: yacs.config.CfgNode
            model configs
        
        checkpoint_path: Path or str
            the path of pretrained model checkpoint, without extension name
        
        Returns
        -------
        DeepSpeech2Model
            The model built from pretrained result.
        """
        model = cls(feat_size=dataset.feature_size,
                    dict_size=dataset.vocab_size,
                    num_conv_layers=config.model.num_conv_layers,
                    num_rnn_layers=config.model.num_rnn_layers,
                    rnn_size=config.model.rnn_layer_size,
                    use_gru=config.model.use_gru,
                    share_rnn_weights=config.model.share_rnn_weights)
        checkpoint.load_parameters(model, checkpoint_path=checkpoint_path)
        layer_tools.summary(model)
        return model


class DeepSpeech2InferModel(DeepSpeech2Model):
    def __init__(self,
                 feat_size,
                 dict_size,
                 num_conv_layers=2,
                 num_rnn_layers=3,
                 rnn_size=1024,
                 use_gru=False,
                 share_rnn_weights=True):
        super().__init__(
            feat_size=feat_size,
            dict_size=dict_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            rnn_size=rnn_size,
            use_gru=use_gru,
            share_rnn_weights=share_rnn_weights)

    def forward(self, audio, audio_len):
        """export model function

        Args:
            audio (Tensor): [B, D, T]
            audio_len (Tensor): [B]

        Returns:
            probs: probs after softmax
        """
        eouts, eouts_len = self.encoder(audio, audio_len)
        probs = self.decoder.probs(eouts)
        return probs
