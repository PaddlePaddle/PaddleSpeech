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
"""Deepspeech2 ASR Model"""
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.fluid.layers import fc
from paddle.nn import GRU
from paddle.nn import LayerList
from paddle.nn import LayerNorm
from paddle.nn import Linear
from paddle.nn import LSTM
from yacs.config import CfgNode

from deepspeech.models.ds2_online.conv import ConvStack
from deepspeech.models.ds2_online.rnn import RNNStack
from deepspeech.modules.ctc import CTCDecoder
from deepspeech.utils import layer_tools
from deepspeech.utils.checkpoint import Checkpoint
from deepspeech.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ['DeepSpeech2ModelOnline', 'DeepSpeech2InferModeOnline']


class CRNNEncoder(nn.Layer):
    def __init__(self,
                 feat_size,
                 dict_size,
                 num_conv_layers=2,
                 num_rnn_layers=4,
                 rnn_size=1024,
                 num_fc_layers=2,
                 fc_layers_size_list=[512, 256],
                 use_gru=False,
                 share_rnn_weights=True):
        super().__init__()
        self.rnn_size = rnn_size
        self.feat_size = feat_size  # 161 for linear
        self.dict_size = dict_size
        self.num_rnn_layers = num_rnn_layers
        self.num_fc_layers = num_fc_layers
        self.fc_layers_size_list = fc_layers_size_list
        self.conv = ConvStack(feat_size, num_conv_layers)

        i_size = self.conv.output_height  # H after conv stack

        self.rnn = LayerList()
        self.layernorm_list = LayerList()
        self.fc_layers_list = LayerList()
        rnn_direction = 'forward'
        layernorm_size = rnn_size

        if use_gru == True:
            self.rnn.append(
                GRU(input_size=i_size,
                    hidden_size=rnn_size,
                    num_layers=1,
                    direction=rnn_direction))
            self.layernorm_list.append(LayerNorm(layernorm_size))
            for i in range(1, num_rnn_layers):
                self.rnn.append(
                    GRU(input_size=layernorm_size,
                        hidden_size=rnn_size,
                        num_layers=1,
                        direction=rnn_direction))
                self.layernorm_list.append(LayerNorm(layernorm_size))
        else:
            self.rnn.append(
                LSTM(
                    input_size=i_size,
                    hidden_size=rnn_size,
                    num_layers=1,
                    direction=rnn_direction))
            self.layernorm_list.append(LayerNorm(layernorm_size))
            for i in range(1, num_rnn_layers):
                self.rnn.append(
                    LSTM(
                        input_size=layernorm_size,
                        hidden_size=rnn_size,
                        num_layers=1,
                        direction=rnn_direction))
                self.layernorm_list.append(LayerNorm(layernorm_size))
        fc_input_size = layernorm_size
        for i in range(self.num_fc_layers):
            self.fc_layers_list.append(
                nn.Linear(fc_input_size, fc_layers_size_list[i]))
            fc_input_size = fc_layers_size_list[i]

    @property
    def output_size(self):
        return self.fc_layers_size_list[-1]

    def forward(self, audio, audio_len):
        """Compute Encoder outputs

        Args:
            audio (Tensor): [B, Tmax, D]
            text (Tensor): [B, Umax]
            audio_len (Tensor): [B]
            text_len (Tensor): [B]
        Returns:
            x (Tensor): encoder outputs, [B, T, D]
            x_lens (Tensor): encoder length, [B]
        """
        # [B, T, D]  -> [B, D, T]
        audio = audio.transpose([0, 2, 1])
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
        x, output_state = self.rnn[0](x, None, x_lens)
        x = self.layernorm_list[0](x)
        for i in range(1, self.num_rnn_layers):
            x, output_state = self.rnn[i](x, output_state, x_lens)  #[B, T, D]
            x = self.layernorm_list[i](x)

        for i in range(self.num_fc_layers):
            x = self.fc_layers_list[i](x)
            x = F.relu(x)
        return x, x_lens


class DeepSpeech2ModelOnline(nn.Layer):
    """The DeepSpeech2 network structure for online.

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
                num_rnn_layers=4,  #Number of stacking RNN layers.
                rnn_layer_size=1024,  #RNN layer size (number of RNN cells).
                num_fc_layers=2,
                fc_layers_size_list=[512, 256],
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
                 num_fc_layers=2,
                 fc_layers_size_list=[512, 256],
                 use_gru=False,
                 share_rnn_weights=True):
        super().__init__()
        self.encoder = CRNNEncoder(
            feat_size=feat_size,
            dict_size=dict_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            num_fc_layers=num_fc_layers,
            fc_layers_size_list=fc_layers_size_list,
            rnn_size=rnn_size,
            use_gru=use_gru,
            share_rnn_weights=share_rnn_weights)
        assert (self.encoder.output_size == fc_layers_size_list[-1])

        self.decoder = CTCDecoder(
            odim=dict_size,  # <blank> is in  vocab
            enc_n_units=self.encoder.output_size,
            blank_id=0,  # first token is <blank>
            dropout_rate=0.0,
            reduction=True,  # sum
            batch_average=True)  # sum / batch_size

    def forward(self, audio, audio_len, text, text_len):
        """Compute Model loss

        Args:
            audio (Tenosr): [B, T, D]
            audio_len (Tensor): [B]
            text (Tensor): [B, U]
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
        probs = self.decoder.softmax(eouts)
        return self.decoder.decode_probs(
            probs.numpy(), eouts_len, vocab_list, decoding_method,
            lang_model_path, beam_alpha, beam_beta, beam_size, cutoff_prob,
            cutoff_top_n, num_processes)

    @classmethod
    def from_pretrained(cls, dataloader, config, checkpoint_path):
        """Build a DeepSpeech2Model model from a pretrained model.
        Parameters
        ----------
        dataloader: paddle.io.DataLoader

        config: yacs.config.CfgNode
            model configs

        checkpoint_path: Path or str
            the path of pretrained model checkpoint, without extension name

        Returns
        -------
        DeepSpeech2Model
            The model built from pretrained result.
        """
        model = cls(feat_size=dataloader.collate_fn.feature_size,
                    dict_size=dataloader.collate_fn.vocab_size,
                    num_conv_layers=config.model.num_conv_layers,
                    num_rnn_layers=config.model.num_rnn_layers,
                    rnn_size=config.model.rnn_layer_size,
                    num_fc_layers=config.model.num_fc_layers,
                    fc_layers_size_list=config.model.fc_layers_size_list,
                    use_gru=config.model.use_gru,
                    share_rnn_weights=config.model.share_rnn_weights)
        infos = Checkpoint().load_parameters(
            model, checkpoint_path=checkpoint_path)
        logger.info(f"checkpoint info: {infos}")
        layer_tools.summary(model)
        return model


class DeepSpeech2InferModelOnline(DeepSpeech2ModelOnline):
    def __init__(self,
                 feat_size,
                 dict_size,
                 num_conv_layers=2,
                 num_rnn_layers=3,
                 rnn_size=1024,
                 num_fc_layers=2,
                 fc_layers_size_list=[512, 256],
                 use_gru=False,
                 share_rnn_weights=True):
        super().__init__(
            feat_size=feat_size,
            dict_size=dict_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            rnn_size=rnn_size,
            num_fc_layers=num_fc_layers,
            fc_layers_size_list=fc_layers_size_list,
            use_gru=use_gru,
            share_rnn_weights=share_rnn_weights)

    def forward(self, audio, audio_len):
        """export model function

        Args:
            audio (Tensor): [B, T, D]
            audio_len (Tensor): [B]

        Returns:
            probs: probs after softmax
        """
        eouts, eouts_len = self.encoder(audio, audio_len)
        probs = self.decoder.softmax(eouts)
        return probs
