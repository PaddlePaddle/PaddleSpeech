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
import numpy as np
import paddle
from paddle import nn

from paddlespeech.t2s.modules.positional_encoding import sinusoid_position_encoding


def expand(encodings: paddle.Tensor, durations: paddle.Tensor) -> paddle.Tensor:
    """
    encodings: (B, T, C)
    durations: (B, T)
    """
    batch_size, t_enc = durations.shape
    durations = durations.numpy()
    slens = np.sum(durations, -1)
    t_dec = np.max(slens)
    M = np.zeros([batch_size, t_dec, t_enc])
    for i in range(batch_size):
        k = 0
        for j in range(t_enc):
            d = durations[i, j]
            M[i, k:k + d, j] = 1
            k += d
    M = paddle.to_tensor(M, dtype=encodings.dtype)
    encodings = paddle.matmul(M, encodings)
    return encodings


class ResidualBlock(nn.Layer):
    def __init__(self, channels, kernel_size, dilation, n=2):
        super().__init__()
        blocks = [
            nn.Sequential(
                nn.Conv1D(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    padding="same",
                    data_format="NLC"),
                nn.ReLU(),
                nn.BatchNorm1D(channels, data_format="NLC"), ) for _ in range(n)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return x + self.blocks(x)


class TextEmbedding(nn.Layer):
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 tone_vocab_size: int=None,
                 tone_embedding_size: int=None,
                 padding_idx: int=None,
                 tone_padding_idx: int=None,
                 concat: bool=False):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, embedding_size,
                                           padding_idx)
        if tone_vocab_size:
            tone_embedding_size = tone_embedding_size or embedding_size
            if tone_embedding_size != embedding_size and not concat:
                raise ValueError(
                    "embedding size != tone_embedding size, only conat is avaiable."
                )
            self.tone_embedding = nn.Embedding(
                tone_vocab_size, tone_embedding_size, tone_padding_idx)
        self.concat = concat

    def forward(self, text, tone=None):
        text_embed = self.text_embedding(text)
        if tone is None:
            return text_embed
        tone_embed = self.tone_embedding(tone)
        if self.concat:
            embed = paddle.concat([text_embed, tone_embed], -1)
        else:
            embed = text_embed + tone_embed
        return embed


class SpeedySpeechEncoder(nn.Layer):
    def __init__(self, vocab_size, tone_size, hidden_size, kernel_size,
                 dilations):
        super().__init__()
        self.embedding = TextEmbedding(
            vocab_size,
            hidden_size,
            tone_size,
            padding_idx=0,
            tone_padding_idx=0)
        self.prenet = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), )
        res_blocks = [
            ResidualBlock(hidden_size, kernel_size, d, n=2) for d in dilations
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        self.postnet1 = nn.Sequential(nn.Linear(hidden_size, hidden_size))
        self.postnet2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1D(hidden_size, data_format="NLC"),
            nn.Linear(hidden_size, hidden_size), )

    def forward(self, text, tones):
        embedding = self.embedding(text, tones)
        embedding = self.prenet(embedding)
        x = self.res_blocks(embedding)
        x = embedding + self.postnet1(x)
        x = self.postnet2(x)
        return x


class DurationPredictor(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualBlock(hidden_size, 4, 1, n=1),
            ResidualBlock(hidden_size, 3, 1, n=1),
            ResidualBlock(hidden_size, 1, 1, n=1), nn.Linear(hidden_size, 1))

    def forward(self, x):
        return paddle.squeeze(self.layers(x), -1)


class SpeedySpeechDecoder(nn.Layer):
    def __init__(self, hidden_size, output_size, kernel_size, dilations):
        super().__init__()
        res_blocks = [
            ResidualBlock(hidden_size, kernel_size, d, n=2) for d in dilations
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        self.postnet1 = nn.Sequential(nn.Linear(hidden_size, hidden_size))
        self.postnet2 = nn.Sequential(
            ResidualBlock(hidden_size, kernel_size, 1, n=2),
            nn.Linear(hidden_size, output_size))

    def forward(self, x):
        xx = self.res_blocks(x)
        x = x + self.postnet1(xx)
        x = self.postnet2(x)
        return x


class SpeedySpeech(nn.Layer):
    def __init__(
            self,
            vocab_size,
            encoder_hidden_size,
            encoder_kernel_size,
            encoder_dilations,
            duration_predictor_hidden_size,
            decoder_hidden_size,
            decoder_output_size,
            decoder_kernel_size,
            decoder_dilations,
            tone_size=None, ):
        super().__init__()
        encoder = SpeedySpeechEncoder(vocab_size, tone_size,
                                      encoder_hidden_size, encoder_kernel_size,
                                      encoder_dilations)
        duration_predictor = DurationPredictor(duration_predictor_hidden_size)
        decoder = SpeedySpeechDecoder(decoder_hidden_size, decoder_output_size,
                                      decoder_kernel_size, decoder_dilations)

        self.encoder = encoder
        self.duration_predictor = duration_predictor
        self.decoder = decoder

    def forward(self, text, tones, durations):
        # input of embedding must be int64
        text = paddle.cast(text, 'int64')
        tones = paddle.cast(tones, 'int64')
        durations = paddle.cast(durations, 'int64')
        encodings = self.encoder(text, tones)
        # (B, T)
        pred_durations = self.duration_predictor(encodings.detach())

        # expand encodings
        durations_to_expand = durations
        encodings = expand(encodings, durations_to_expand)

        # decode
        # remove positional encoding here
        _, t_dec, feature_size = encodings.shape
        encodings += sinusoid_position_encoding(t_dec, feature_size)
        decoded = self.decoder(encodings)
        return decoded, pred_durations

    def inference(self, text, tones=None):
        # text: [T]
        # tones: [T]
        # input of embedding must be int64
        text = paddle.cast(text, 'int64')
        text = text.unsqueeze(0)
        if tones is not None:
            tones = paddle.cast(tones, 'int64')
            tones = tones.unsqueeze(0)

        encodings = self.encoder(text, tones)
        pred_durations = self.duration_predictor(encodings)  # (1, T)
        durations_to_expand = paddle.round(pred_durations.exp())
        durations_to_expand = (durations_to_expand).astype(paddle.int64)

        slens = paddle.sum(durations_to_expand, -1)  # [1]
        t_dec = slens[0]  # [1]
        t_enc = paddle.shape(pred_durations)[-1]
        M = paddle.zeros([1, t_dec, t_enc])

        k = paddle.full([1], 0, dtype=paddle.int64)
        for j in range(t_enc):
            d = durations_to_expand[0, j]
            # If the d == 0, slice action is meaningless and not supported
            if d >= 1:
                M[0, k:k + d, j] = 1
            k += d

        encodings = paddle.matmul(M, encodings)

        shape = paddle.shape(encodings)
        t_dec, feature_size = shape[1], shape[2]
        encodings += sinusoid_position_encoding(t_dec, feature_size)
        decoded = self.decoder(encodings)
        return decoded[0]


class SpeedySpeechInference(nn.Layer):
    def __init__(self, normalizer, speedyspeech_model):
        super().__init__()
        self.normalizer = normalizer
        self.acoustic_model = speedyspeech_model

    def forward(self, phones, tones):
        normalized_mel = self.acoustic_model.inference(phones, tones)
        logmel = self.normalizer.inverse(normalized_mel)
        return logmel
