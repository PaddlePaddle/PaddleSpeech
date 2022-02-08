# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import nn
from paddle.fluid.layers import sequence_mask
from paddle.nn import functional as F
from paddle.nn import initializer as I
from tqdm import trange

from paddlespeech.t2s.modules.conv import Conv1dBatchNorm
from paddlespeech.t2s.modules.losses import guided_attention_loss
from paddlespeech.t2s.utils import checkpoint

__all__ = ["Tacotron2", "Tacotron2Loss"]


class LocationSensitiveAttention(nn.Layer):
    """Location Sensitive Attention module.

    Reference: `Attention-Based Models for Speech Recognition <https://arxiv.org/pdf/1506.07503.pdf>`_

    Parameters
    -----------
    d_query: int
        The feature size of query.
    d_key : int
        The feature size of key.
    d_attention : int
        The feature size of dimension.
    location_filters : int
        Filter size of attention convolution.
    location_kernel_size : int
        Kernel size of attention convolution.
    """

    def __init__(self,
                 d_query: int,
                 d_key: int,
                 d_attention: int,
                 location_filters: int,
                 location_kernel_size: int):
        super().__init__()

        self.query_layer = nn.Linear(d_query, d_attention, bias_attr=False)
        self.key_layer = nn.Linear(d_key, d_attention, bias_attr=False)
        self.value = nn.Linear(d_attention, 1, bias_attr=False)

        # Location Layer
        self.location_conv = nn.Conv1D(
            2,
            location_filters,
            kernel_size=location_kernel_size,
            padding=int((location_kernel_size - 1) / 2),
            bias_attr=False,
            data_format='NLC')
        self.location_layer = nn.Linear(
            location_filters, d_attention, bias_attr=False)

    def forward(self,
                query,
                processed_key,
                value,
                attention_weights_cat,
                mask=None):
        """Compute context vector and attention weights.
        
        Parameters
        -----------
        query : Tensor [shape=(batch_size, d_query)]
            The queries.
        processed_key : Tensor [shape=(batch_size, time_steps_k, d_attention)]
            The keys after linear layer.
        value : Tensor [shape=(batch_size, time_steps_k, d_key)]
            The values.
        attention_weights_cat : Tensor [shape=(batch_size, time_step_k, 2)]
            Attention weights concat.
        mask : Tensor, optional
            The mask. Shape should be (batch_size, times_steps_k, 1).
            Defaults to None.

        Returns
        ----------
        attention_context : Tensor [shape=(batch_size, d_attention)]
            The context vector.
        attention_weights : Tensor [shape=(batch_size, time_steps_k)]
            The attention weights.
        """

        processed_query = self.query_layer(paddle.unsqueeze(query, axis=[1]))
        processed_attention_weights = self.location_layer(
            self.location_conv(attention_weights_cat))
        # (B, T_enc, 1)
        alignment = self.value(
            paddle.tanh(processed_attention_weights + processed_key +
                        processed_query))

        if mask is not None:
            alignment = alignment + (1.0 - mask) * -1e9

        attention_weights = F.softmax(alignment, axis=1)
        attention_context = paddle.matmul(
            attention_weights, value, transpose_x=True)

        attention_weights = paddle.squeeze(attention_weights, axis=-1)
        attention_context = paddle.squeeze(attention_context, axis=1)

        return attention_context, attention_weights


class DecoderPreNet(nn.Layer):
    """Decoder prenet module for Tacotron2.

    Parameters
    ----------
    d_input: int
        The input feature size.

    d_hidden: int
        The hidden size.

    d_output: int
        The output feature size.

    dropout_rate: float
        The droput probability.

    """

    def __init__(self,
                 d_input: int,
                 d_hidden: int,
                 d_output: int,
                 dropout_rate: float):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(d_input, d_hidden, bias_attr=False)
        self.linear2 = nn.Linear(d_hidden, d_output, bias_attr=False)

    def forward(self, x):
        """Calculate forward propagation.

        Parameters
        ----------
        x: Tensor [shape=(B, T_mel, C)]
            Batch of the sequences of padded mel spectrogram.

        Returns
        -------
        output: Tensor [shape=(B, T_mel, C)]
            Batch of the sequences of padded hidden state.

        """

        x = F.dropout(F.relu(self.linear1(x)), self.dropout_rate, training=True)
        output = F.dropout(
            F.relu(self.linear2(x)), self.dropout_rate, training=True)
        return output


class DecoderPostNet(nn.Layer):
    """Decoder postnet module for Tacotron2.

    Parameters
    ----------
    d_mels: int
        The number of mel bands.

    d_hidden: int
        The hidden size of postnet.

    kernel_size: int
        The kernel size of the conv layer in postnet.

    num_layers: int
        The number of conv layers in postnet.

    dropout: float
        The droput probability.

    """

    def __init__(self,
                 d_mels: int,
                 d_hidden: int,
                 kernel_size: int,
                 num_layers: int,
                 dropout: float):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        padding = int((kernel_size - 1) / 2)

        self.conv_batchnorms = nn.LayerList()
        k = math.sqrt(1.0 / (d_mels * kernel_size))
        self.conv_batchnorms.append(
            Conv1dBatchNorm(
                d_mels,
                d_hidden,
                kernel_size=kernel_size,
                padding=padding,
                bias_attr=I.Uniform(-k, k),
                data_format='NLC'))

        k = math.sqrt(1.0 / (d_hidden * kernel_size))
        self.conv_batchnorms.extend([
            Conv1dBatchNorm(
                d_hidden,
                d_hidden,
                kernel_size=kernel_size,
                padding=padding,
                bias_attr=I.Uniform(-k, k),
                data_format='NLC') for i in range(1, num_layers - 1)
        ])

        self.conv_batchnorms.append(
            Conv1dBatchNorm(
                d_hidden,
                d_mels,
                kernel_size=kernel_size,
                padding=padding,
                bias_attr=I.Uniform(-k, k),
                data_format='NLC'))

    def forward(self, x):
        """Calculate forward propagation.

        Parameters
        ----------
        x: Tensor [shape=(B, T_mel, C)]
            Output sequence of features from decoder.

        Returns
        -------
        output: Tensor [shape=(B, T_mel, C)]
            Output sequence of features after postnet.

        """

        for i in range(len(self.conv_batchnorms) - 1):
            x = F.dropout(
                F.tanh(self.conv_batchnorms[i](x)),
                self.dropout,
                training=self.training)
        output = F.dropout(
            self.conv_batchnorms[self.num_layers - 1](x),
            self.dropout,
            training=self.training)
        return output


class Tacotron2Encoder(nn.Layer):
    """Tacotron2 encoder module for Tacotron2.

    Parameters
    ----------
    d_hidden: int
        The hidden size in encoder module.

    conv_layers: int
        The number of conv layers.

    kernel_size: int
        The kernel size of conv layers.

    p_dropout: float
        The droput probability.
    """

    def __init__(self,
                 d_hidden: int,
                 conv_layers: int,
                 kernel_size: int,
                 p_dropout: float):
        super().__init__()

        k = math.sqrt(1.0 / (d_hidden * kernel_size))
        self.conv_batchnorms = nn.LayerList([
            Conv1dBatchNorm(
                d_hidden,
                d_hidden,
                kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                bias_attr=I.Uniform(-k, k),
                data_format='NLC') for i in range(conv_layers)
        ])
        self.p_dropout = p_dropout

        self.hidden_size = int(d_hidden / 2)
        self.lstm = nn.LSTM(
            d_hidden, self.hidden_size, direction="bidirectional")

    def forward(self, x, input_lens=None):
        """Calculate forward propagation of tacotron2 encoder.

        Parameters
        ----------
        x: Tensor [shape=(B, T, C)]
            Input embeddings.

        text_lens: Tensor [shape=(B,)], optional
            Batch of lengths of each text input batch. Defaults to None.

        Returns
        -------
        output : Tensor [shape=(B, T, C)]
            Batch of the sequences of padded hidden states.

        """
        for conv_batchnorm in self.conv_batchnorms:
            x = F.dropout(
                F.relu(conv_batchnorm(x)),
                self.p_dropout,
                training=self.training)

        output, _ = self.lstm(inputs=x, sequence_length=input_lens)
        return output


class Tacotron2Decoder(nn.Layer):
    """Tacotron2 decoder module for Tacotron2.

    Parameters
    ----------
    d_mels: int
        The number of mel bands.

    reduction_factor: int
        The reduction factor of tacotron.

    d_encoder: int
        The hidden size of encoder.

    d_prenet: int
        The hidden size in decoder prenet.

    d_attention_rnn: int
        The attention rnn layer hidden size.

    d_decoder_rnn: int
        The decoder rnn layer hidden size.

    d_attention: int
        The hidden size of the linear layer in location sensitive attention.

    attention_filters: int
        The filter size of the conv layer in location sensitive attention.

    attention_kernel_size: int
        The kernel size of the conv layer in location sensitive attention.

    p_prenet_dropout: float
        The droput probability in decoder prenet.

    p_attention_dropout: float
        The droput probability in location sensitive attention.

    p_decoder_dropout: float
        The droput probability in decoder.

    use_stop_token: bool
        Whether to use a binary classifier for stop token prediction. 
        Defaults to False
    """

    def __init__(self,
                 d_mels: int,
                 reduction_factor: int,
                 d_encoder: int,
                 d_prenet: int,
                 d_attention_rnn: int,
                 d_decoder_rnn: int,
                 d_attention: int,
                 attention_filters: int,
                 attention_kernel_size: int,
                 p_prenet_dropout: float,
                 p_attention_dropout: float,
                 p_decoder_dropout: float,
                 use_stop_token: bool=False):
        super().__init__()
        self.d_mels = d_mels
        self.reduction_factor = reduction_factor
        self.d_encoder = d_encoder
        self.d_attention_rnn = d_attention_rnn
        self.d_decoder_rnn = d_decoder_rnn
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout

        self.prenet = DecoderPreNet(
            d_mels * reduction_factor,
            d_prenet,
            d_prenet,
            dropout_rate=p_prenet_dropout)

        # attention_rnn takes attention's context vector has an
        # auxiliary input
        self.attention_rnn = nn.LSTMCell(d_prenet + d_encoder, d_attention_rnn)

        self.attention_layer = LocationSensitiveAttention(
            d_attention_rnn, d_encoder, d_attention, attention_filters,
            attention_kernel_size)

        # decoder_rnn takes prenet's output and attention_rnn's input
        # as input
        self.decoder_rnn = nn.LSTMCell(d_attention_rnn + d_encoder,
                                       d_decoder_rnn)
        self.linear_projection = nn.Linear(d_decoder_rnn + d_encoder,
                                           d_mels * reduction_factor)

        self.use_stop_token = use_stop_token
        if use_stop_token:
            self.stop_layer = nn.Linear(d_decoder_rnn + d_encoder, 1)

        # states - temporary attributes
        self.attention_hidden = None
        self.attention_cell = None

        self.decoder_hidden = None
        self.decoder_cell = None

        self.attention_weights = None
        self.attention_weights_cum = None
        self.attention_context = None

        self.key = None
        self.mask = None
        self.processed_key = None

    def _initialize_decoder_states(self, key):
        """init states be used in decoder
        """
        batch_size, encoder_steps, _ = key.shape

        self.attention_hidden = paddle.zeros(
            shape=[batch_size, self.d_attention_rnn], dtype=key.dtype)
        self.attention_cell = paddle.zeros(
            shape=[batch_size, self.d_attention_rnn], dtype=key.dtype)

        self.decoder_hidden = paddle.zeros(
            shape=[batch_size, self.d_decoder_rnn], dtype=key.dtype)
        self.decoder_cell = paddle.zeros(
            shape=[batch_size, self.d_decoder_rnn], dtype=key.dtype)

        self.attention_weights = paddle.zeros(
            shape=[batch_size, encoder_steps], dtype=key.dtype)
        self.attention_weights_cum = paddle.zeros(
            shape=[batch_size, encoder_steps], dtype=key.dtype)
        self.attention_context = paddle.zeros(
            shape=[batch_size, self.d_encoder], dtype=key.dtype)

        self.key = key  # [B, T, C]
        # pre-compute projected keys to improve efficiency
        self.processed_key = self.attention_layer.key_layer(key)  # [B, T, C]

    def _decode(self, query):
        """decode one time step
        """
        cell_input = paddle.concat([query, self.attention_context], axis=-1)

        # The first lstm layer (or spec encoder lstm)
        _, (self.attention_hidden, self.attention_cell) = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden,
            self.p_attention_dropout,
            training=self.training)

        # Loaction sensitive attention
        attention_weights_cat = paddle.stack(
            [self.attention_weights, self.attention_weights_cum], axis=-1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.processed_key, self.key,
            attention_weights_cat, self.mask)
        self.attention_weights_cum += self.attention_weights

        # The second lstm layer (or spec decoder lstm)
        decoder_input = paddle.concat(
            [self.attention_hidden, self.attention_context], axis=-1)
        _, (self.decoder_hidden, self.decoder_cell) = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden,
            p=self.p_decoder_dropout,
            training=self.training)

        # decode output one step
        decoder_hidden_attention_context = paddle.concat(
            [self.decoder_hidden, self.attention_context], axis=-1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        if self.use_stop_token:
            stop_logit = self.stop_layer(decoder_hidden_attention_context)
            return decoder_output, self.attention_weights, stop_logit
        return decoder_output, self.attention_weights

    def forward(self, keys, querys, mask):
        """Calculate forward propagation of tacotron2 decoder.

        Parameters
        ----------
        keys: Tensor[shape=(B, T_key, C)]
            Batch of the sequences of padded output from encoder.

        querys: Tensor[shape(B, T_query, C)]
            Batch of the sequences of padded mel spectrogram.

        mask: Tensor
            Mask generated with text length. Shape should be (B, T_key, 1).

        Returns
        -------
        mel_output: Tensor [shape=(B, T_query, C)]
            Output sequence of features.

        alignments: Tensor [shape=(B, T_query, T_key)]
            Attention weights.
        """
        self._initialize_decoder_states(keys)
        self.mask = mask

        querys = paddle.reshape(
            querys,
            [querys.shape[0], querys.shape[1] // self.reduction_factor, -1])
        start_step = paddle.zeros(
            shape=[querys.shape[0], 1, querys.shape[-1]], dtype=querys.dtype)
        querys = paddle.concat([start_step, querys], axis=1)

        querys = self.prenet(querys)

        mel_outputs, alignments = [], []
        stop_logits = []
        # Ignore the last time step
        while len(mel_outputs) < querys.shape[1] - 1:
            query = querys[:, len(mel_outputs), :]
            if self.use_stop_token:
                mel_output, attention_weights, stop_logit = self._decode(query)
            else:
                mel_output, attention_weights = self._decode(query)
            mel_outputs.append(mel_output)
            alignments.append(attention_weights)
            if self.use_stop_token:
                stop_logits.append(stop_logit)

        alignments = paddle.stack(alignments, axis=1)
        mel_outputs = paddle.stack(mel_outputs, axis=1)
        if self.use_stop_token:
            stop_logits = paddle.concat(stop_logits, axis=1)
            return mel_outputs, alignments, stop_logits
        return mel_outputs, alignments

    def infer(self, key, max_decoder_steps=1000):
        """Calculate forward propagation of tacotron2 decoder.

        Parameters
        ----------
        keys: Tensor [shape=(B, T_key, C)]
            Batch of the sequences of padded output from encoder.

        max_decoder_steps: int, optional
            Number of max step when synthesize. Defaults to 1000.

        Returns
        -------
        mel_output: Tensor [shape=(B, T_mel, C)]
            Output sequence of features.

        alignments: Tensor [shape=(B, T_mel, T_key)]
            Attention weights.

        """
        self._initialize_decoder_states(key)
        self.mask = None  # mask is not needed for single instance inference
        encoder_steps = key.shape[1]

        # [B, C]
        start_step = paddle.zeros(
            shape=[key.shape[0], self.d_mels * self.reduction_factor],
            dtype=key.dtype)
        query = start_step  # [B, C]
        first_hit_end = None

        mel_outputs, alignments = [], []
        stop_logits = []
        for i in trange(max_decoder_steps):
            query = self.prenet(query)
            if self.use_stop_token:
                mel_output, alignment, stop_logit = self._decode(query)
            else:
                mel_output, alignment = self._decode(query)

            mel_outputs.append(mel_output)
            alignments.append(alignment)  # (B=1, T)
            if self.use_stop_token:
                stop_logits.append(stop_logit)

            if self.use_stop_token:
                if F.sigmoid(stop_logit) > 0.5:
                    print("hit stop condition!")
                    break
            else:
                if int(paddle.argmax(alignment[0])) == encoder_steps - 1:
                    if first_hit_end is None:
                        first_hit_end = i
                    elif i > (first_hit_end + 20):
                        print("content exhausted!")
                        break
            if len(mel_outputs) == max_decoder_steps:
                print("Warning! Reached max decoder steps!!!")
                break

            query = mel_output

        alignments = paddle.stack(alignments, axis=1)
        mel_outputs = paddle.stack(mel_outputs, axis=1)
        if self.use_stop_token:
            stop_logits = paddle.concat(stop_logits, axis=1)
            return mel_outputs, alignments, stop_logits
        return mel_outputs, alignments


class Tacotron2(nn.Layer):
    """Tacotron2 model for end-to-end text-to-speech (E2E-TTS).

    This is a model of Spectrogram prediction network in Tacotron2 described
    in `Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram
    Predictions <https://arxiv.org/abs/1712.05884>`_,
    which converts the sequence of characters
    into the sequence of mel spectrogram.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size of phons of the model.

    n_tones: int
        Vocabulary size of tones of the model. Defaults to None. If provided,
        the model has an extra tone embedding.

    d_mels: int
        Number of mel bands.

    d_encoder: int
        Hidden size in encoder module.

    encoder_conv_layers: int
        Number of conv layers in encoder.

    encoder_kernel_size: int
        Kernel size of conv layers in encoder.

    d_prenet: int
        Hidden size in decoder prenet.

    d_attention_rnn: int
        Attention rnn layer hidden size in decoder.

    d_decoder_rnn: int
        Decoder rnn layer hidden size in decoder.

    attention_filters: int
        Filter size of the conv layer in location sensitive attention.

    attention_kernel_size: int
        Kernel size of the conv layer in location sensitive attention.

    d_attention: int
        Hidden size of the linear layer in location sensitive attention.

    d_postnet: int
        Hidden size of postnet.

    postnet_kernel_size: int
        Kernel size of the conv layer in postnet.

    postnet_conv_layers: int
        Number of conv layers in postnet.

    reduction_factor: int
        Reduction factor of tacotron2.

    p_encoder_dropout: float
        Droput probability in encoder.

    p_prenet_dropout: float
        Droput probability in decoder prenet.

    p_attention_dropout: float
        Droput probability in location sensitive attention.

    p_decoder_dropout: float
        Droput probability in decoder.

    p_postnet_dropout: float
        Droput probability in postnet.

    d_global_condition: int
        Feature size of global condition. Defaults to None. If provided, The 
        model assumes a global condition that is concatenated to the encoder
        outputs.

    """

    def __init__(self,
                 vocab_size,
                 n_tones=None,
                 d_mels: int=80,
                 d_encoder: int=512,
                 encoder_conv_layers: int=3,
                 encoder_kernel_size: int=5,
                 d_prenet: int=256,
                 d_attention_rnn: int=1024,
                 d_decoder_rnn: int=1024,
                 attention_filters: int=32,
                 attention_kernel_size: int=31,
                 d_attention: int=128,
                 d_postnet: int=512,
                 postnet_kernel_size: int=5,
                 postnet_conv_layers: int=5,
                 reduction_factor: int=1,
                 p_encoder_dropout: float=0.5,
                 p_prenet_dropout: float=0.5,
                 p_attention_dropout: float=0.1,
                 p_decoder_dropout: float=0.1,
                 p_postnet_dropout: float=0.5,
                 d_global_condition=None,
                 use_stop_token=False):
        super().__init__()

        std = math.sqrt(2.0 / (vocab_size + d_encoder))
        val = math.sqrt(3.0) * std  # uniform bounds for std
        self.embedding = nn.Embedding(
            vocab_size, d_encoder, weight_attr=I.Uniform(-val, val))
        if n_tones:
            self.embedding_tones = nn.Embedding(
                n_tones,
                d_encoder,
                padding_idx=0,
                weight_attr=I.Uniform(-0.1 * val, 0.1 * val))
        self.toned = n_tones is not None

        self.encoder = Tacotron2Encoder(d_encoder, encoder_conv_layers,
                                        encoder_kernel_size, p_encoder_dropout)

        # input augmentation scheme: concat global condition to the encoder output
        if d_global_condition is not None:
            d_encoder += d_global_condition
        self.decoder = Tacotron2Decoder(
            d_mels,
            reduction_factor,
            d_encoder,
            d_prenet,
            d_attention_rnn,
            d_decoder_rnn,
            d_attention,
            attention_filters,
            attention_kernel_size,
            p_prenet_dropout,
            p_attention_dropout,
            p_decoder_dropout,
            use_stop_token=use_stop_token)
        self.postnet = DecoderPostNet(
            d_mels=d_mels * reduction_factor,
            d_hidden=d_postnet,
            kernel_size=postnet_kernel_size,
            num_layers=postnet_conv_layers,
            dropout=p_postnet_dropout)

    def forward(self,
                text_inputs,
                text_lens,
                mels,
                output_lens=None,
                tones=None,
                global_condition=None):
        """Calculate forward propagation of tacotron2.

        Parameters
        ----------
        text_inputs: Tensor [shape=(B, T_text)]
            Batch of the sequencees of padded character ids.

        text_lens: Tensor [shape=(B,)]
            Batch of lengths of each text input batch.

        mels: Tensor [shape(B, T_mel, C)]
            Batch of the sequences of padded mel spectrogram.

        output_lens: Tensor [shape=(B,)], optional
            Batch of lengths of each mels batch. Defaults to None.

        tones: Tensor [shape=(B, T_text)]
            Batch of sequences of padded tone ids.

        global_condition: Tensor [shape(B, C)]
            Batch of global conditions. Defaults to None. If the 
            `d_global_condition` of the model is not None, this input should be
            provided.

        use_stop_token: bool
            Whether to include a binary classifier to predict the stop token. 
            Defaults to False.
            
        Returns
        -------
        outputs : Dict[str, Tensor]

            mel_output: output sequence of features (B, T_mel, C);

            mel_outputs_postnet: output sequence of features after postnet (B, T_mel, C);

            alignments: attention weights (B, T_mel, T_text);

            stop_logits: output sequence of stop logits (B, T_mel)
        """
        # input of embedding must be int64
        text_inputs = paddle.cast(text_inputs, 'int64')
        embedded_inputs = self.embedding(text_inputs)
        if self.toned:
            embedded_inputs += self.embedding_tones(tones)

        encoder_outputs = self.encoder(embedded_inputs, text_lens)

        if global_condition is not None:
            global_condition = global_condition.unsqueeze(1)
            global_condition = paddle.expand(global_condition,
                                             [-1, encoder_outputs.shape[1], -1])
            encoder_outputs = paddle.concat([encoder_outputs, global_condition],
                                            -1)

        # [B, T_enc, 1]
        mask = sequence_mask(
            text_lens, dtype=encoder_outputs.dtype).unsqueeze(-1)
        if self.decoder.use_stop_token:
            mel_outputs, alignments, stop_logits = self.decoder(
                encoder_outputs, mels, mask=mask)
        else:
            mel_outputs, alignments = self.decoder(
                encoder_outputs, mels, mask=mask)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if output_lens is not None:
            # [B, T_dec, 1]
            mask = sequence_mask(output_lens).unsqueeze(-1)
            mel_outputs = mel_outputs * mask  # [B, T, C]
            mel_outputs_postnet = mel_outputs_postnet * mask  # [B, T, C]
        outputs = {
            "mel_output": mel_outputs,
            "mel_outputs_postnet": mel_outputs_postnet,
            "alignments": alignments
        }
        if self.decoder.use_stop_token:
            outputs["stop_logits"] = stop_logits

        return outputs

    @paddle.no_grad()
    def infer(self,
              text_inputs,
              max_decoder_steps=1000,
              tones=None,
              global_condition=None):
        """Generate the mel sepctrogram of features given the sequences of character ids.

        Parameters
        ----------
        text_inputs: Tensor [shape=(B, T_text)]
            Batch of the sequencees of padded character ids.

        max_decoder_steps: int, optional
            Number of max step when synthesize. Defaults to 1000.

        Returns
        -------
        outputs : Dict[str, Tensor]

            mel_output: output sequence of sepctrogram (B, T_mel, C);

            mel_outputs_postnet: output sequence of sepctrogram after postnet (B, T_mel, C);

            stop_logits: output sequence of stop logits (B, T_mel);

            alignments: attention weights (B, T_mel, T_text). This key is only
            present when `use_stop_token` is True.
        """
        # input of embedding must be int64
        text_inputs = paddle.cast(text_inputs, 'int64')
        embedded_inputs = self.embedding(text_inputs)
        if self.toned:
            embedded_inputs += self.embedding_tones(tones)
        encoder_outputs = self.encoder(embedded_inputs)

        if global_condition is not None:
            global_condition = global_condition.unsqueeze(1)
            global_condition = paddle.expand(global_condition,
                                             [-1, encoder_outputs.shape[1], -1])
            encoder_outputs = paddle.concat([encoder_outputs, global_condition],
                                            -1)
        if self.decoder.use_stop_token:
            mel_outputs, alignments, stop_logits = self.decoder.infer(
                encoder_outputs, max_decoder_steps=max_decoder_steps)
        else:
            mel_outputs, alignments = self.decoder.infer(
                encoder_outputs, max_decoder_steps=max_decoder_steps)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = {
            "mel_output": mel_outputs,
            "mel_outputs_postnet": mel_outputs_postnet,
            "alignments": alignments
        }
        if self.decoder.use_stop_token:
            outputs["stop_logits"] = stop_logits

        return outputs

    @classmethod
    def from_pretrained(cls, config, checkpoint_path):
        """Build a Tacotron2 model from a pretrained model.

        Parameters
        ----------
        config: yacs.config.CfgNode
            model configs

        checkpoint_path: Path or str
            the path of pretrained model checkpoint, without extension name

        Returns
        -------
        ConditionalWaveFlow
            The model built from pretrained result.
        """
        model = cls(vocab_size=config.model.vocab_size,
                    n_tones=config.model.n_tones,
                    d_mels=config.data.n_mels,
                    d_encoder=config.model.d_encoder,
                    encoder_conv_layers=config.model.encoder_conv_layers,
                    encoder_kernel_size=config.model.encoder_kernel_size,
                    d_prenet=config.model.d_prenet,
                    d_attention_rnn=config.model.d_attention_rnn,
                    d_decoder_rnn=config.model.d_decoder_rnn,
                    attention_filters=config.model.attention_filters,
                    attention_kernel_size=config.model.attention_kernel_size,
                    d_attention=config.model.d_attention,
                    d_postnet=config.model.d_postnet,
                    postnet_kernel_size=config.model.postnet_kernel_size,
                    postnet_conv_layers=config.model.postnet_conv_layers,
                    reduction_factor=config.model.reduction_factor,
                    p_encoder_dropout=config.model.p_encoder_dropout,
                    p_prenet_dropout=config.model.p_prenet_dropout,
                    p_attention_dropout=config.model.p_attention_dropout,
                    p_decoder_dropout=config.model.p_decoder_dropout,
                    p_postnet_dropout=config.model.p_postnet_dropout,
                    d_global_condition=config.model.d_global_condition,
                    use_stop_token=config.model.use_stop_token)
        checkpoint.load_parameters(model, checkpoint_path=checkpoint_path)
        return model


class Tacotron2Loss(nn.Layer):
    """ Tacotron2 Loss module
    """

    def __init__(self,
                 use_stop_token_loss=True,
                 use_guided_attention_loss=False,
                 sigma=0.2):
        """Tacotron 2 Criterion.

        Args:
            use_stop_token_loss (bool, optional): Whether to use a loss for stop token prediction. Defaults to True.
            use_guided_attention_loss (bool, optional): Whether to use a loss for attention weights. Defaults to False.
            sigma (float, optional): Hyper-parameter sigma for guided attention loss. Defaults to 0.2.
        """
        super().__init__()
        self.spec_criterion = nn.MSELoss()
        self.use_stop_token_loss = use_stop_token_loss
        self.use_guided_attention_loss = use_guided_attention_loss
        self.attn_criterion = guided_attention_loss
        self.stop_criterion = nn.BCEWithLogitsLoss()
        self.sigma = sigma

    def forward(self,
                mel_outputs,
                mel_outputs_postnet,
                mel_targets,
                attention_weights=None,
                slens=None,
                plens=None,
                stop_logits=None):
        """Calculate tacotron2 loss.

        Parameters
        ----------
        mel_outputs: Tensor [shape=(B, T_mel, C)]
            Output mel spectrogram sequence.

        mel_outputs_postnet: Tensor [shape(B, T_mel, C)]
            Output mel spectrogram sequence after postnet.

        mel_targets: Tensor [shape=(B, T_mel, C)]
            Target mel spectrogram sequence.

        attention_weights: Tensor [shape=(B, T_mel, T_enc)]
            Attention weights. This should be provided when 
            `use_guided_attention_loss` is True.
        
        slens: Tensor [shape=(B,)]
            Number of frames of mel spectrograms. This should be provided when 
            `use_guided_attention_loss` is True.
        
        plens: Tensor [shape=(B, )]
            Number of text or phone ids of each utterance. This should be 
            provided when `use_guided_attention_loss` is True.

        stop_logits: Tensor [shape=(B, T_mel)]
            Stop logits of each mel spectrogram frame. This should be provided 
            when `use_stop_token_loss` is True.

        Returns
        -------
        losses : Dict[str, Tensor]

            loss: the sum of the other three losses;

            mel_loss: MSE loss compute by mel_targets and mel_outputs;

            post_mel_loss: MSE loss compute by mel_targets and mel_outputs_postnet;

            guided_attn_loss: Guided attention loss for attention weights;

            stop_loss: Binary cross entropy loss for stop token prediction.
        """
        mel_loss = self.spec_criterion(mel_outputs, mel_targets)
        post_mel_loss = self.spec_criterion(mel_outputs_postnet, mel_targets)
        total_loss = mel_loss + post_mel_loss
        if self.use_guided_attention_loss:
            gal_loss = self.attn_criterion(attention_weights, slens, plens,
                                           self.sigma)
            total_loss += gal_loss
        if self.use_stop_token_loss:
            T_dec = mel_targets.shape[1]
            stop_labels = F.one_hot(slens - 1, num_classes=T_dec)
            stop_token_loss = self.stop_criterion(stop_logits, stop_labels)
            total_loss += stop_token_loss

        losses = {
            "loss": total_loss,
            "mel_loss": mel_loss,
            "post_mel_loss": post_mel_loss
        }
        if self.use_guided_attention_loss:
            losses["guided_attn_loss"] = gal_loss
        if self.use_stop_token_loss:
            losses["stop_loss"] = stop_token_loss
        return losses
