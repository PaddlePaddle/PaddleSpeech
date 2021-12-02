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
from paddle import nn
from yacs.config import CfgNode

from paddlespeech.s2t.models.ds2.conv import ConvStack
from paddlespeech.s2t.models.ds2.rnn import RNNStack
from paddlespeech.s2t.modules.ctc import CTCDecoder
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils.checkpoint import Checkpoint
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ['DeepSpeech2Model', 'DeepSpeech2InferModel']


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
                share_rnn_weights=True,  #Whether to share input-hidden weights between forward and backward directional RNNs.Notice that for GRU, weight sharing is not supported.
                ctc_grad_norm_type='instance', ))
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
                 share_rnn_weights=True,
                 blank_id=0,
                 ctc_grad_norm_type='instance'):
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
            odim=dict_size,  # <blank> is in  vocab
            enc_n_units=self.encoder.output_size,
            blank_id=blank_id,
            dropout_rate=0.0,
            reduction=True,  # sum
            batch_average=True,  # sum / batch_size
            grad_norm_type=ctc_grad_norm_type)

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
        model = cls(
            feat_size=dataloader.collate_fn.feature_size,
            #feat_size=dataloader.dataset.feature_size,
            dict_size=dataloader.collate_fn.vocab_size,
            #dict_size=dataloader.dataset.vocab_size,
            num_conv_layers=config.model.num_conv_layers,
            num_rnn_layers=config.model.num_rnn_layers,
            rnn_size=config.model.rnn_layer_size,
            use_gru=config.model.use_gru,
            share_rnn_weights=config.model.share_rnn_weights,
            blank_id=config.model.blank_id,
            ctc_grad_norm_type=config.model.ctc_grad_norm_type, )
        infos = Checkpoint().load_parameters(
            model, checkpoint_path=checkpoint_path)
        logger.info(f"checkpoint info: {infos}")
        layer_tools.summary(model)
        return model

    @classmethod
    def from_config(cls, config):
        """Build a DeepSpeec2Model from config
        Parameters

        config: yacs.config.CfgNode
            config.model
        Returns
        -------
        DeepSpeech2Model
            The model built from config.
        """
        model = cls(
            feat_size=config.input_dim,
            dict_size=config.output_dim,
            num_conv_layers=config.num_conv_layers,
            num_rnn_layers=config.num_rnn_layers,
            rnn_size=config.rnn_layer_size,
            use_gru=config.use_gru,
            share_rnn_weights=config.share_rnn_weights,
            blank_id=config.blank_id,
            ctc_grad_norm_type=config.ctc_grad_norm_type, )
        return model


class DeepSpeech2InferModel(DeepSpeech2Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        return probs, eouts_len

    def export(self):
        static_model = paddle.jit.to_static(
            self,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, None, self.encoder.feat_size],
                    dtype='float32'),  # audio, [B,T,D]
                paddle.static.InputSpec(shape=[None],
                                        dtype='int64'),  # audio_length, [B]
            ])
        return static_model
