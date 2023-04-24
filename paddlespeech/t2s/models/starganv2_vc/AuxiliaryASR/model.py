# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from paddle import nn

from .layers import Attention
from .layers import ConvBlock
from .layers import ConvNorm
from .layers import LinearNorm
from .layers import MFCC
from paddlespeech.t2s.modules.nets_utils import _reset_parameters
from paddlespeech.utils.initialize import uniform_


class ASRCNN(nn.Layer):
    def __init__(
            self,
            input_dim: int=80,
            hidden_dim: int=256,
            n_token: int=35,
            n_layers: int=6,
            token_embedding_dim: int=256, ):
        super().__init__()
        self.n_token = n_token
        self.n_down = 1
        self.to_mfcc = MFCC()
        self.init_cnn = ConvNorm(
            in_channels=input_dim // 2,
            out_channels=hidden_dim,
            kernel_size=7,
            padding=3,
            stride=2)
        self.cnns = nn.Sequential(* [
            nn.Sequential(
                ConvBlock(hidden_dim),
                nn.GroupNorm(num_groups=1, num_channels=hidden_dim))
            for n in range(n_layers)
        ])
        self.projection = ConvNorm(
            in_channels=hidden_dim, out_channels=hidden_dim // 2)
        self.ctc_linear = nn.Sequential(
            LinearNorm(in_dim=hidden_dim // 2, out_dim=hidden_dim),
            nn.ReLU(), LinearNorm(in_dim=hidden_dim, out_dim=n_token))
        self.asr_s2s = ASRS2S(
            embedding_dim=token_embedding_dim,
            hidden_dim=hidden_dim // 2,
            n_token=n_token)

        self.reset_parameters()
        self.asr_s2s.reset_parameters()

    def forward(self,
                x: paddle.Tensor,
                src_key_padding_mask: paddle.Tensor=None,
                text_input: paddle.Tensor=None):
        x = self.to_mfcc(x)
        x = self.init_cnn(x)
        x = self.cnns(x)
        x = self.projection(x)
        x = x.transpose([0, 2, 1])
        ctc_logit = self.ctc_linear(x)
        if text_input is not None:
            _, s2s_logit, s2s_attn = self.asr_s2s(
                memory=x,
                memory_mask=src_key_padding_mask,
                text_input=text_input)
            return ctc_logit, s2s_logit, s2s_attn
        else:
            return ctc_logit

    def get_feature(self, x: paddle.Tensor):
        x = self.to_mfcc(x.squeeze(1))
        x = self.init_cnn(x)
        x = self.cnns(x)
        x = self.projection(x)
        return x

    def length_to_mask(self, lengths: paddle.Tensor):
        mask = paddle.arange(lengths.max()).unsqueeze(0).expand(
            (lengths.shape[0], -1)).astype(lengths.dtype)
        mask = paddle.greater_than(mask + 1, lengths.unsqueeze(1))
        return mask

    def get_future_mask(self, out_length: int, unmask_future_steps: int=0):
        """
        Args:
            out_length (int):
                returned mask shape is (out_length, out_length).
            unmask_futre_steps (int): 
                unmasking future step size.
        Return:
            Tensor (paddle.Tensor(bool)): 
                mask future timesteps mask[i, j] = True if i > j + unmask_future_steps else False
        """
        index_tensor = paddle.arange(out_length).unsqueeze(0).expand(
            [out_length, -1])
        mask = paddle.greater_than(index_tensor,
                                   index_tensor.T + unmask_future_steps)
        return mask

    def reset_parameters(self):
        self.apply(_reset_parameters)


class ASRS2S(nn.Layer):
    def __init__(self,
                 embedding_dim: int=256,
                 hidden_dim: int=512,
                 n_location_filters: int=32,
                 location_kernel_size: int=63,
                 n_token: int=40):
        super().__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        self.val_range = math.sqrt(6 / hidden_dim)

        self.decoder_rnn_dim = hidden_dim
        self.project_to_n_symbols = nn.Linear(self.decoder_rnn_dim, n_token)
        self.attention_layer = Attention(
            attention_rnn_dim=self.decoder_rnn_dim,
            embedding_dim=hidden_dim,
            attention_dim=hidden_dim,
            attention_location_n_filters=n_location_filters,
            attention_location_kernel_size=location_kernel_size)
        self.decoder_rnn = nn.LSTMCell(self.decoder_rnn_dim + embedding_dim,
                                       self.decoder_rnn_dim)
        self.project_to_hidden = nn.Sequential(
            LinearNorm(in_dim=self.decoder_rnn_dim * 2, out_dim=hidden_dim),
            nn.Tanh())
        self.sos = 1
        self.eos = 2

    def initialize_decoder_states(self,
                                  memory: paddle.Tensor,
                                  mask: paddle.Tensor):
        """
        moemory.shape = (B, L, H) = (Batchsize, Maxtimestep, Hiddendim)
        """
        B, L, H = memory.shape
        dtype = memory.dtype
        self.decoder_hidden = paddle.zeros(
            (B, self.decoder_rnn_dim)).astype(dtype)
        self.decoder_cell = paddle.zeros(
            (B, self.decoder_rnn_dim)).astype(dtype)
        self.attention_weights = paddle.zeros((B, L)).astype(dtype)
        self.attention_weights_cum = paddle.zeros((B, L)).astype(dtype)
        self.attention_context = paddle.zeros((B, H)).astype(dtype)
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
        self.unk_index = 3
        self.random_mask = 0.1

    def forward(self,
                memory: paddle.Tensor,
                memory_mask: paddle.Tensor,
                text_input: paddle.Tensor):
        """
        moemory.shape = (B, L, H) = (Batchsize, Maxtimestep, Hiddendim)
        moemory_mask.shape = (B, L, )
        texts_input.shape = (B, T)
        """
        self.initialize_decoder_states(memory, memory_mask)
        # text random mask
        random_mask = (paddle.rand(text_input.shape) < self.random_mask)
        _text_input = text_input.clone()
        _text_input[:] = paddle.where(
            condition=random_mask,
            x=paddle.full(
                shape=_text_input.shape,
                fill_value=self.unk_index,
                dtype=_text_input.dtype),
            y=_text_input)
        decoder_inputs = self.embedding(_text_input).transpose(
            [1, 0, 2])  # -> [T, B, channel]
        start_embedding = self.embedding(
            paddle.to_tensor(
                [self.sos] * decoder_inputs.shape[1], dtype=paddle.long))
        decoder_inputs = paddle.concat(
            (start_embedding.unsqueeze(0), decoder_inputs), axis=0)

        hidden_outputs, logit_outputs, alignments = [], [], []
        while len(hidden_outputs) < decoder_inputs.shape[0]:
            decoder_input = decoder_inputs[len(hidden_outputs)]
            hidden, logit, attention_weights = self.decode(decoder_input)
            hidden_outputs += [hidden]
            logit_outputs += [logit]
            alignments += [attention_weights]

        hidden_outputs, logit_outputs, alignments = self.parse_decoder_outputs(
            hidden_outputs, logit_outputs, alignments)

        return hidden_outputs, logit_outputs, alignments

    def decode(self, decoder_input: paddle.Tensor):
        cell_input = paddle.concat((decoder_input, self.attention_context), -1)
        self.decoder_rnn.flatten_parameters()
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            cell_input, (self.decoder_hidden, self.decoder_cell))

        attention_weights_cat = paddle.concat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)),
            axis=1)

        self.attention_context, self.attention_weights = self.attention_layer(
            self.decoder_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights

        hidden_and_context = paddle.concat(
            (self.decoder_hidden, self.attention_context), -1)
        hidden = self.project_to_hidden(hidden_and_context)

        # dropout to increasing g
        logit = self.project_to_n_symbols(F.dropout(hidden, 0.5, self.training))

        return hidden, logit, self.attention_weights

    def parse_decoder_outputs(self,
                              hidden: paddle.Tensor,
                              logit: paddle.Tensor,
                              alignments: paddle.Tensor):
        # -> [B, T_out + 1, max_time]
        alignments = paddle.stack(alignments).transpose([1, 0, 2])
        # [T_out + 1, B, n_symbols] -> [B, T_out + 1,  n_symbols]
        logit = paddle.stack(logit).transpose([1, 0, 2])
        hidden = paddle.stack(hidden).transpose([1, 0, 2])

        return hidden, logit, alignments

    def reset_parameters(self):
        uniform_(self.embedding.weight, -self.val_range, self.val_range)
