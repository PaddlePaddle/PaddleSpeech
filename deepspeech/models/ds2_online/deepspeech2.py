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
"""Deepspeech2 ASR Online Model"""
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import nn
from yacs.config import CfgNode

from deepspeech.models.ds2_online.conv import Conv2dSubsampling4Online
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
                 rnn_direction='forward',
                 num_fc_layers=2,
                 fc_layers_size_list=[512, 256],
                 use_gru=False):
        super().__init__()
        self.rnn_size = rnn_size
        self.feat_size = feat_size  # 161 for linear
        self.dict_size = dict_size
        self.num_rnn_layers = num_rnn_layers
        self.num_fc_layers = num_fc_layers
        self.rnn_direction = rnn_direction
        self.fc_layers_size_list = fc_layers_size_list
        self.conv = Conv2dSubsampling4Online(feat_size, 32, dropout_rate=0.0)

        i_size = self.conv.output_dim

        self.rnn = nn.LayerList()
        self.layernorm_list = nn.LayerList()
        self.fc_layers_list = nn.LayerList()
        layernorm_size = rnn_size
        for i in range(0, num_rnn_layers):
            if i == 0:
                rnn_input_size = i_size
            else:
                rnn_input_size = rnn_size
            if (use_gru == True):
                self.rnn.append(
                    nn.GRU(
                        input_size=rnn_input_size,
                        hidden_size=rnn_size,
                        num_layers=1,
                        direction=rnn_direction))
            else:
                self.rnn.append(
                    nn.LSTM(
                        input_size=rnn_input_size,
                        hidden_size=rnn_size,
                        num_layers=1,
                        direction=rnn_direction))
            self.layernorm_list.append(nn.LayerNorm(layernorm_size))

        fc_input_size = rnn_size
        for i in range(self.num_fc_layers):
            self.fc_layers_list.append(
                nn.Linear(fc_input_size, fc_layers_size_list[i]))
            fc_input_size = fc_layers_size_list[i]

    @property
    def output_size(self):
        return self.fc_layers_size_list[-1]

    def forward(self, x, x_lens):
        """Compute Encoder outputs

        Args:
            x (Tensor): [B, T_input, D]
            x_lens (Tensor): [B]
        Returns:
            x (Tensor): encoder outputs, [B, T_output, D]
            x_lens (Tensor): encoder length, [B]
            final_state_list: list of final_states for RNN layers, [num_directions, batch_size, hidden_size] * num_rnn_layers
        """
        # [B, T, D]
        # convolution group
        x, x_lens = self.conv(x, x_lens)
        # convert data from convolution feature map to sequence of vectors
        #B, C, D, T = paddle.shape(x)  # not work under jit
        #x = x.transpose([0, 3, 1, 2])  #[B, T, C, D]
        #x = x.reshape([B, T, C * D])  #[B, T, C*D]  # not work under jit
        #x = x.reshape([0, 0, -1])  #[B, T, C*D]

        # remove padding part
        init_state = None
        final_state_list = []
        for i in range(0, self.num_rnn_layers):
            x, final_state = self.rnn[i](x, init_state, x_lens)  #[B, T, D]
            final_state_list.append(final_state)
            x = self.layernorm_list[i](x)

        for i in range(self.num_fc_layers):
            x = self.fc_layers_list[i](x)
            x = F.relu(x)
        return x, x_lens, final_state_list

    def forward_chunk(self, x, x_lens, init_state_list):
        """Compute Encoder outputs

        Args:
            x (Tensor): [B, feature_chunk_size, D]
            x_lens (Tensor): [B]
            init_state_list (list of Tensors): [ num_directions, batch_size, hidden_size] * num_rnn_layers
        Returns:
            x (Tensor): encoder outputs, [B, chunk_size, D]
            x_lens (Tensor): encoder length, [B]
            chunk_final_state_list: list of final_states for RNN layers, [num_directions, batch_size, hidden_size] * num_rnn_layers
        """
        x, x_lens = self.conv(x, x_lens)
        chunk_final_state_list = []
        for i in range(0, self.num_rnn_layers):
            x, final_state = self.rnn[i](x, init_state_list[i],
                                         x_lens)  #[B, T, D]
            chunk_final_state_list.append(final_state)
            x = self.layernorm_list[i](x)

        for i in range(self.num_fc_layers):
            x = self.fc_layers_list[i](x)
            x = F.relu(x)
        return x, x_lens, chunk_final_state_list

    def forward_chunk_by_chunk(self, x, x_lens, decoder_chunk_size=8):
        subsampling_rate = self.conv.subsampling_rate
        receptive_field_length = self.conv.receptive_field_length
        chunk_size = (decoder_chunk_size - 1
                      ) * subsampling_rate + receptive_field_length
        chunk_stride = subsampling_rate * decoder_chunk_size
        max_len = x.shape[1]
        assert (chunk_size <= max_len)

        eouts_chunk_list = []
        eouts_chunk_lens_list = []

        padding_len = chunk_stride - (max_len - chunk_size) % chunk_stride
        padding = paddle.zeros((x.shape[0], padding_len, x.shape[2]))
        padded_x = paddle.concat([x, padding], axis=1)
        num_chunk = (max_len + padding_len - chunk_size) / chunk_stride + 1
        num_chunk = int(num_chunk)
        chunk_state_list = [None] * self.num_rnn_layers
        for i in range(0, num_chunk):
            start = i * chunk_stride
            end = start + chunk_size
            #   end = min(start + chunk_size, max_len)
            #   if (end - start < receptive_field_length):
            #       break
            x_chunk = padded_x[:, start:end, :]

            x_len_left = paddle.where(x_lens - i * chunk_stride < 0,
                                      paddle.zeros_like(x_lens),
                                      x_lens - i * chunk_stride)
            x_chunk_len_tmp = paddle.ones_like(x_lens) * chunk_size
            x_chunk_lens = paddle.where(x_len_left < x_chunk_len_tmp,
                                        x_len_left, x_chunk_len_tmp)

            eouts_chunk, eouts_chunk_lens, chunk_state_list = self.forward_chunk(
                x_chunk, x_chunk_lens, chunk_state_list)

            eouts_chunk_list.append(eouts_chunk)
            eouts_chunk_lens_list.append(eouts_chunk_lens)

        return eouts_chunk_list, eouts_chunk_lens_list, chunk_state_list


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
                 rnn_direction='forward',
                 num_fc_layers=2,
                 fc_layers_size_list=[512, 256],
                 use_gru=False):
        super().__init__()
        self.encoder = CRNNEncoder(
            feat_size=feat_size,
            dict_size=dict_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            rnn_direction=rnn_direction,
            num_fc_layers=num_fc_layers,
            fc_layers_size_list=fc_layers_size_list,
            rnn_size=rnn_size,
            use_gru=use_gru)
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
        eouts, eouts_len, final_state_list = self.encoder(audio, audio_len)
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

        eouts, eouts_len, final_state_list = self.encoder(audio, audio_len)
        probs = self.decoder.softmax(eouts)
        return self.decoder.decode_probs(
            probs.numpy(), eouts_len, vocab_list, decoding_method,
            lang_model_path, beam_alpha, beam_beta, beam_size, cutoff_prob,
            cutoff_top_n, num_processes)
    """
    @paddle.no_grad()
    def decode_by_chunk(self, eouts_prefix, eouts_len_prefix, chunk_state_list,
                        audio_chunk, audio_len_chunk, vocab_list,
                        decoding_method, lang_model_path, beam_alpha, beam_beta,
                        beam_size, cutoff_prob, cutoff_top_n, num_processes):
        # init once
        # decoders only accept string encoded in utf-8
        self.decoder.init_decode(
            beam_alpha=beam_alpha,
            beam_beta=beam_beta,
            lang_model_path=lang_model_path,
            vocab_list=vocab_list,
            decoding_method=decoding_method)

        eouts_chunk, eouts_chunk_len, final_state_list = self.encoder.forward_chunk(
            audio_chunk, audio_len_chunk, chunk_state_list)
        if eouts_prefix is not None:
            eouts = paddle.concat([eouts_prefix, eouts_chunk], axis=1)
            eouts_len = paddle.add_n([eouts_len_prefix, eouts_chunk_len])
        else:
            eouts = eouts_chunk
            eouts_len = eouts_chunk_len

        probs = self.decoder.softmax(eouts)
        return self.decoder.decode_probs(
            probs.numpy(), eouts_len, vocab_list, decoding_method,
            lang_model_path, beam_alpha, beam_beta, beam_size, cutoff_prob,
            cutoff_top_n, num_processes), eouts, eouts_len, final_state_list

    @paddle.no_grad()
    def decode_chunk_by_chunk(self, audio, audio_len, vocab_list,
                              decoding_method, lang_model_path, beam_alpha,
                              beam_beta, beam_size, cutoff_prob, cutoff_top_n,
                              num_processes):
        # init once
        # decoders only accept string encoded in utf-8
        self.decoder.init_decode(
            beam_alpha=beam_alpha,
            beam_beta=beam_beta,
            lang_model_path=lang_model_path,
            vocab_list=vocab_list,
            decoding_method=decoding_method)

        eouts_chunk_list, eouts_chunk_len_list, final_state_list = self.encoder.forward_chunk_by_chunk(
            audio, audio_len)
        eouts = paddle.concat(eouts_chunk_list, axis=1)
        eouts_len = paddle.add_n(eouts_chunk_len_list)

        probs = self.decoder.softmax(eouts)
        return self.decoder.decode_probs(
            probs.numpy(), eouts_len, vocab_list, decoding_method,
            lang_model_path, beam_alpha, beam_beta, beam_size, cutoff_prob,
            cutoff_top_n, num_processes)
    """
    """
    decocd_prob,
    decode_prob_chunk_by_chunk
    decode_prob_by_chunk
    is only used for test
    """
    """
    @paddle.no_grad()
    def decode_prob(self, audio, audio_len):
        eouts, eouts_len, final_state_list = self.encoder(audio, audio_len)
        probs = self.decoder.softmax(eouts)
        return probs, eouts, eouts_len, final_state_list

    @paddle.no_grad()
    def decode_prob_chunk_by_chunk(self, audio, audio_len, decoder_chunk_size):
        eouts_chunk_list, eouts_chunk_len_list, final_state_list = self.encoder.forward_chunk_by_chunk(
            audio, audio_len, decoder_chunk_size)
        eouts = paddle.concat(eouts_chunk_list, axis=1)
        eouts_len = paddle.add_n(eouts_chunk_len_list)
        probs = self.decoder.softmax(eouts)
        return probs, eouts, eouts_len, final_state_list

    @paddle.no_grad()
    def decode_prob_by_chunk(self, audio, audio_len, eouts_prefix,
                             eouts_lens_prefix, chunk_state_list):
        eouts_chunk, eouts_chunk_lens, final_state_list = self.encoder.forward_chunk(
            audio, audio_len, chunk_state_list)
        if eouts_prefix is not None:
            eouts = paddle.concat([eouts_prefix, eouts_chunk], axis=1)
            eouts_lens = paddle.add_n([eouts_lens_prefix, eouts_chunk_lens])
        else:
            eouts = eouts_chunk
            eouts_lens = eouts_chunk_lens
        probs = self.decoder.softmax(eouts)
        return probs, eouts, eouts_lens, final_state_list
    """

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
                    rnn_direction=config.model.rnn_direction,
                    num_fc_layers=config.model.num_fc_layers,
                    fc_layers_size_list=config.model.fc_layers_size_list,
                    use_gru=config.model.use_gru)
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
                 rnn_direction='forward',
                 num_fc_layers=2,
                 fc_layers_size_list=[512, 256],
                 use_gru=False):
        super().__init__(
            feat_size=feat_size,
            dict_size=dict_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            rnn_size=rnn_size,
            rnn_direction=rnn_direction,
            num_fc_layers=num_fc_layers,
            fc_layers_size_list=fc_layers_size_list,
            use_gru=use_gru)

    def forward(self, audio, audio_len):
        """export model function

        Args:
            audio (Tensor): [B, T, D]
            audio_len (Tensor): [B]

        Returns:
            probs: probs after softmax
        """
        eouts, eouts_len, final_state_list = self.encoder(audio, audio_len)
        probs = self.decoder.softmax(eouts)
        return probs

    def forward_chunk(self, audio_chunk, audio_chunk_lens):
        eouts_chunkt, eouts_chunk_lens, final_state_list = self.encoder.forward_chunk(
            audio_chunk, audio_chunk_lens)
        probs = self.decoder.softmax(eouts)
        return probs

    def forward(self, eouts_chunk_prefix, eouts_chunk_lens_prefix, audio_chunk,
                audio_chunk_lens, chunk_state_list):
        """export model function

        Args:
            audio_chunk (Tensor): [B, T, D]
            audio_chunk_len (Tensor): [B]

        Returns:
            probs: probs after softmax
        """
        eouts_chunk, eouts_chunk_lens, final_state_list = self.encoder.forward_chunk(
            audio_chunk, audio_chunk_lens, chunk_state_list)
        eouts_chunk_new_prefix = paddle.concat(
            [eouts_chunk_prefix, eouts_chunk], axis=1)
        eouts_chunk_lens_new_prefix = paddle.add(eouts_chunk_lens_prefix,
                                                 eouts_chunk_lens)
        probs_chunk = self.decoder.softmax(eouts_chunk_new_prefix)
        return probs_chunk, eouts_chunk_new_prefix, eouts_chunk_lens_new_prefix, final_state_list
