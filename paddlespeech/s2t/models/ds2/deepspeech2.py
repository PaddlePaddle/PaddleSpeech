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
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.s2t.models.ds2.conv import Conv2dSubsampling4Pure
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
        self.use_gru = use_gru
        self.conv = Conv2dSubsampling4Pure(feat_size, 32, dropout_rate=0.0)

        self.output_dim = self.conv.output_dim

        i_size = self.conv.output_dim
        self.rnn = nn.LayerList()
        self.layernorm_list = nn.LayerList()
        self.fc_layers_list = nn.LayerList()
        if rnn_direction == 'bidirect' or rnn_direction == 'bidirectional':
            layernorm_size = 2 * rnn_size
        elif rnn_direction == 'forward':
            layernorm_size = rnn_size
        else:
            raise Exception("Wrong rnn direction")
        for i in range(0, num_rnn_layers):
            if i == 0:
                rnn_input_size = i_size
            else:
                rnn_input_size = layernorm_size
            if use_gru is True:
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
            self.output_dim = layernorm_size

        fc_input_size = layernorm_size
        for i in range(self.num_fc_layers):
            self.fc_layers_list.append(
                nn.Linear(fc_input_size, fc_layers_size_list[i]))
            fc_input_size = fc_layers_size_list[i]
            self.output_dim = fc_layers_size_list[i]

    @property
    def output_size(self):
        return self.output_dim

    def forward(self, x, x_lens, init_state_h_box=None, init_state_c_box=None):
        """Compute Encoder outputs

        Args:
            x (Tensor): [B, T, D]
            x_lens (Tensor): [B]
            init_state_h_box(Tensor): init_states h for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
            init_state_c_box(Tensor): init_states c for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
        Return:
            x (Tensor): encoder outputs, [B, T, D]
            x_lens (Tensor): encoder length, [B]
            final_state_h_box(Tensor): final_states h for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
            final_state_c_box(Tensor): final_states c for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
        """
        if init_state_h_box is not None:
            init_state_list = None

            if self.use_gru is True:
                init_state_h_list = paddle.split(
                    init_state_h_box, self.num_rnn_layers, axis=0)
                init_state_list = init_state_h_list
            else:
                init_state_h_list = paddle.split(
                    init_state_h_box, self.num_rnn_layers, axis=0)
                init_state_c_list = paddle.split(
                    init_state_c_box, self.num_rnn_layers, axis=0)
                init_state_list = [(init_state_h_list[i], init_state_c_list[i])
                                   for i in range(self.num_rnn_layers)]
        else:
            init_state_list = [None] * self.num_rnn_layers

        x, x_lens = self.conv(x, x_lens)
        final_chunk_state_list = []
        for i in range(0, self.num_rnn_layers):
            x, final_state = self.rnn[i](x, init_state_list[i],
                                         x_lens)  #[B, T, D]
            final_chunk_state_list.append(final_state)
            x = self.layernorm_list[i](x)

        for i in range(self.num_fc_layers):
            x = self.fc_layers_list[i](x)
            x = F.relu(x)

        if self.use_gru is True:
            final_chunk_state_h_box = paddle.concat(
                final_chunk_state_list, axis=0)
            final_chunk_state_c_box = init_state_c_box
        else:
            final_chunk_state_h_list = [
                final_chunk_state_list[i][0] for i in range(self.num_rnn_layers)
            ]
            final_chunk_state_c_list = [
                final_chunk_state_list[i][1] for i in range(self.num_rnn_layers)
            ]
            final_chunk_state_h_box = paddle.concat(
                final_chunk_state_h_list, axis=0)
            final_chunk_state_c_box = paddle.concat(
                final_chunk_state_c_list, axis=0)

        return x, x_lens, final_chunk_state_h_box, final_chunk_state_c_box

    def forward_chunk_by_chunk(self, x, x_lens, decoder_chunk_size=8):
        """Compute Encoder outputs

        Args:
            x (Tensor): [B, T, D]
            x_lens (Tensor): [B]
            decoder_chunk_size: The chunk size of decoder
        Returns:
            eouts_list (List of Tensor): The list of encoder outputs in chunk_size: [B, chunk_size, D] * num_chunks
            eouts_lens_list (List of Tensor): The list of  encoder length in chunk_size: [B] * num_chunks
            final_state_h_box(Tensor): final_states h for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
            final_state_c_box(Tensor): final_states c for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
        """
        subsampling_rate = self.conv.subsampling_rate
        receptive_field_length = self.conv.receptive_field_length
        chunk_size = (decoder_chunk_size - 1
                      ) * subsampling_rate + receptive_field_length
        chunk_stride = subsampling_rate * decoder_chunk_size
        max_len = x.shape[1]
        assert (chunk_size <= max_len)

        eouts_chunk_list = []
        eouts_chunk_lens_list = []
        if (max_len - chunk_size) % chunk_stride != 0:
            padding_len = chunk_stride - (max_len - chunk_size) % chunk_stride
        else:
            padding_len = 0
        padding = paddle.zeros((x.shape[0], padding_len, x.shape[2]))
        padded_x = paddle.concat([x, padding], axis=1)
        num_chunk = (max_len + padding_len - chunk_size) / chunk_stride + 1
        num_chunk = int(num_chunk)
        chunk_state_h_box = None
        chunk_state_c_box = None
        final_state_h_box = None
        final_state_c_box = None
        for i in range(0, num_chunk):
            start = i * chunk_stride
            end = start + chunk_size
            x_chunk = padded_x[:, start:end, :]

            x_len_left = paddle.where(x_lens - i * chunk_stride < 0,
                                      paddle.zeros_like(x_lens),
                                      x_lens - i * chunk_stride)
            x_chunk_len_tmp = paddle.ones_like(x_lens) * chunk_size
            x_chunk_lens = paddle.where(x_len_left < x_chunk_len_tmp,
                                        x_len_left, x_chunk_len_tmp)

            eouts_chunk, eouts_chunk_lens, chunk_state_h_box, chunk_state_c_box = self.forward(
                x_chunk, x_chunk_lens, chunk_state_h_box, chunk_state_c_box)

            eouts_chunk_list.append(eouts_chunk)
            eouts_chunk_lens_list.append(eouts_chunk_lens)
        final_state_h_box = chunk_state_h_box
        final_state_c_box = chunk_state_c_box
        return eouts_chunk_list, eouts_chunk_lens_list, final_state_h_box, final_state_c_box


class DeepSpeech2Model(nn.Layer):
    """The DeepSpeech2 network structure.

    :param audio: Audio spectrogram data layer.
    :type audio: Variable
    :param text: Transcription text data layer.
    :type text: Variable
    :param audio_len: Valid sequence length data layer.
    :type audio_len: Variable
    :param feat_size: feature size for audio.
    :type feat_size: int
    :param dict_size: Dictionary size for tokenized transcription.
    :type dict_size: int
    :param num_conv_layers: Number of stacking convolution layers.
    :type num_conv_layers: int
    :param num_rnn_layers: Number of stacking RNN layers.
    :type num_rnn_layers: int
    :param rnn_size: RNN layer size (dimension of RNN cells).
    :type rnn_size: int
    :param num_fc_layers: Number of stacking FC layers.
    :type num_fc_layers: int
    :param fc_layers_size_list: The list of FC layer sizes.
    :type fc_layers_size_list: [int,]
    :param use_gru: Use gru if set True. Use simple rnn if set False.
    :type use_gru: bool
    :return: A tuple of an output unnormalized log probability layer (
             before softmax) and a ctc cost layer.
    :rtype: tuple of LayerOutput
    """

    def __init__(
            self,
            feat_size,
            dict_size,
            num_conv_layers=2,
            num_rnn_layers=4,
            rnn_size=1024,
            rnn_direction='forward',
            num_fc_layers=2,
            fc_layers_size_list=[512, 256],
            use_gru=False,
            blank_id=0,
            ctc_grad_norm_type=None, ):
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
            audio (Tensor): [B, T, D]
            audio_len (Tensor): [B]
            text (Tensor): [B, U]
            text_len (Tensor): [B]

        Returns:
            loss (Tensor): [1]
        """
        eouts, eouts_len, final_state_h_box, final_state_c_box = self.encoder(
            audio, audio_len, None, None)
        loss = self.decoder(eouts, eouts_len, text, text_len)
        return loss

    @paddle.no_grad()
    def decode(self, audio, audio_len):
        # decoders only accept string encoded in utf-8
        # Make sure the decoder has been initialized
        eouts, eouts_len, final_state_h_box, final_state_c_box = self.encoder(
            audio, audio_len, None, None)
        probs = self.decoder.softmax(eouts)
        batch_size = probs.shape[0]
        self.decoder.reset_decoder(batch_size=batch_size)
        self.decoder.next(probs, eouts_len)
        trans_best, trans_beam = self.decoder.decode()
        return trans_best

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
            feat_size=dataloader.feat_dim,
            dict_size=dataloader.vocab_size,
            num_conv_layers=config.num_conv_layers,
            num_rnn_layers=config.num_rnn_layers,
            rnn_size=config.rnn_layer_size,
            rnn_direction=config.rnn_direction,
            num_fc_layers=config.num_fc_layers,
            fc_layers_size_list=config.fc_layers_size_list,
            use_gru=config.use_gru,
            blank_id=config.blank_id,
            ctc_grad_norm_type=config.get('ctc_grad_norm_type', None), )
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
            config
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
            rnn_direction=config.rnn_direction,
            num_fc_layers=config.num_fc_layers,
            fc_layers_size_list=config.fc_layers_size_list,
            use_gru=config.use_gru,
            blank_id=config.blank_id,
            ctc_grad_norm_type=config.get('ctc_grad_norm_type', None), )
        return model


class DeepSpeech2InferModel(DeepSpeech2Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                audio_chunk,
                audio_chunk_lens,
                chunk_state_h_box=None,
                chunk_state_c_box=None):
        if self.encoder.rnn_direction == "forward":
            eouts_chunk, eouts_chunk_lens, final_state_h_box, final_state_c_box = self.encoder(
                audio_chunk, audio_chunk_lens, chunk_state_h_box,
                chunk_state_c_box)
            probs_chunk = self.decoder.softmax(eouts_chunk)
            return probs_chunk, eouts_chunk_lens, final_state_h_box, final_state_c_box
        elif self.encoder.rnn_direction == "bidirect":
            eouts, eouts_len, _, _ = self.encoder(audio_chunk, audio_chunk_lens)
            probs = self.decoder.softmax(eouts)
            return probs, eouts_len
        else:
            raise Exception("wrong model type")

    def export(self):
        if self.encoder.rnn_direction == "forward":
            static_model = paddle.jit.to_static(
                self,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None, None, self.encoder.feat_size
                               ],  #[B, chunk_size, feat_dim]
                        dtype='float32', ),
                    paddle.static.InputSpec(shape=[None],
                                            dtype='int64'),  # audio_length, [B]
                    paddle.static.InputSpec(
                        shape=[None, None, None], dtype='float32'),
                    paddle.static.InputSpec(
                        shape=[None, None, None], dtype='float32')
                ],
                full_graph=True)
        elif self.encoder.rnn_direction == "bidirect":
            static_model = paddle.jit.to_static(
                self,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None, None, self.encoder.feat_size],
                        dtype='float32'),  # audio, [B,T,D]
                    paddle.static.InputSpec(shape=[None],
                                            dtype='int64'),  # audio_length, [B]
                ],
                full_graph=True)
        else:
            raise Exception("wrong model type")
        return static_model
