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
import paddle
import paddle.nn.functional as F
import paddleaudio.functional as audio_F
from paddle import nn

from paddlespeech.utils.initialize import _calculate_gain
from paddlespeech.utils.initialize import xavier_uniform_


def _get_activation_fn(activ):
    if activ == 'relu':
        return nn.ReLU()
    elif activ == 'lrelu':
        return nn.LeakyReLU(0.2)
    elif activ == 'swish':
        return nn.Swish()
    else:
        raise RuntimeError(
            'Unexpected activ type %s, expected [relu, lrelu, swish]' % activ)


class LinearNorm(nn.Layer):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 bias: bool=True,
                 w_init_gain: str='linear'):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias_attr=bias)
        xavier_uniform_(
            self.linear_layer.weight, gain=_calculate_gain(w_init_gain))

    def forward(self, x: paddle.Tensor):
        out = self.linear_layer(x)
        return out


class ConvNorm(nn.Layer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int=1,
                 stride: int=1,
                 padding: int=None,
                 dilation: int=1,
                 bias: bool=True,
                 w_init_gain: str='linear',
                 param=None):
        super().__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias_attr=bias)

        xavier_uniform_(
            self.conv.weight, gain=_calculate_gain(w_init_gain, param=param))

    def forward(self, signal: paddle.Tensor):
        conv_signal = self.conv(signal)
        return conv_signal


class ConvBlock(nn.Layer):
    def __init__(self,
                 hidden_dim: int,
                 n_conv: int=3,
                 dropout_p: float=0.2,
                 activ: str='relu'):
        super().__init__()
        self._n_groups = 8
        self.blocks = nn.LayerList([
            self._get_conv(
                hidden_dim=hidden_dim,
                dilation=3**i,
                activ=activ,
                dropout_p=dropout_p) for i in range(n_conv)
        ])

    def forward(self, x: paddle.Tensor):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self,
                  hidden_dim: int,
                  dilation: int,
                  activ: str='relu',
                  dropout_p: float=0.2):
        layers = [
            ConvNorm(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation), _get_activation_fn(activ),
            nn.GroupNorm(num_groups=self._n_groups, num_channels=hidden_dim),
            nn.Dropout(p=dropout_p), ConvNorm(
                hidden_dim, hidden_dim, kernel_size=3, padding=1,
                dilation=1), _get_activation_fn(activ), nn.Dropout(p=dropout_p)
        ]
        return nn.Sequential(*layers)


class LocationLayer(nn.Layer):
    def __init__(self,
                 attention_n_filters: int,
                 attention_kernel_size: int,
                 attention_dim: int):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1)
        self.location_dense = LinearNorm(
            in_dim=attention_n_filters,
            out_dim=attention_dim,
            bias=False,
            w_init_gain='tanh')

    def forward(self, attention_weights_cat: paddle.Tensor):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose([0, 2, 1])
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Layer):
    def __init__(self,
                 attention_rnn_dim: int,
                 embedding_dim: int,
                 attention_dim: int,
                 attention_location_n_filters: int,
                 attention_location_kernel_size: int):
        super().__init__()
        self.query_layer = LinearNorm(
            in_dim=attention_rnn_dim,
            out_dim=attention_dim,
            bias=False,
            w_init_gain='tanh')
        self.memory_layer = LinearNorm(
            in_dim=embedding_dim,
            out_dim=attention_dim,
            bias=False,
            w_init_gain='tanh')
        self.v = LinearNorm(in_dim=attention_dim, out_dim=1, bias=False)
        self.location_layer = LocationLayer(
            attention_n_filters=attention_location_n_filters,
            attention_kernel_size=attention_location_kernel_size,
            attention_dim=attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self,
                               query: paddle.Tensor,
                               processed_memory: paddle.Tensor,
                               attention_weights_cat: paddle.Tensor):
        """
        Args:
            query: 
                decoder output (B, n_mel_channels * n_frames_per_step)
            processed_memory: 
                processed encoder outputs (B, T_in, attention_dim)
            attention_weights_cat: 
                cumulative and prev. att weights (B, 2, max_time)
        Returns:
            Tensor: 
                alignment (B, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            paddle.tanh(processed_query + processed_attention_weights +
                        processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self,
                attention_hidden_state: paddle.Tensor,
                memory: paddle.Tensor,
                processed_memory: paddle.Tensor,
                attention_weights_cat: paddle.Tensor,
                mask: paddle.Tensor):
        """
        Args:
            attention_hidden_state: 
                attention rnn last output
            memory: 
                encoder outputs
            processed_memory: 
                processed encoder outputs
            attention_weights_cat: 
                previous and cummulative attention weights
            mask: 
                binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            query=attention_hidden_state,
            processed_memory=processed_memory,
            attention_weights_cat=attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, axis=1)
        attention_context = paddle.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class MFCC(nn.Layer):
    def __init__(self, n_mfcc: int=40, n_mels: int=80):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.norm = 'ortho'
        dct_mat = audio_F.create_dct(self.n_mfcc, self.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)

    def forward(self, mel_specgram: paddle.Tensor):
        if len(mel_specgram.shape) == 2:
            mel_specgram = mel_specgram.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False
        # (channel, n_mels, time).tranpose(...) dot (n_mels, n_mfcc)
        # -> (channel, time, n_mfcc).tranpose(...)
        mfcc = paddle.matmul(mel_specgram.transpose([0, 2, 1]),
                             self.dct_mat).transpose([0, 2, 1])
        # unpack batch
        if unsqueezed:
            mfcc = mfcc.squeeze(0)
        return mfcc
