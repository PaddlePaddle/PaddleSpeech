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
from typing import List

import paddle
from paddle import nn

from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.predictor.length_regulator import LengthRegulator
from paddlespeech.t2s.modules.transformer.embedding import ScaledPositionalEncoding


class ResidualBlock(nn.Layer):
    def __init__(self,
                 channels: int=128,
                 kernel_size: int=3,
                 dilation: int=3,
                 n: int=2):
        """SpeedySpeech encoder module.
        Args:
            channels (int, optional): Feature size of the resiaudl output(and also the input).
            kernel_size (int, optional): Kernel size of the 1D convolution.
            dilation (int, optional): Dilation of the 1D convolution.
            n (int): Number of blocks.
        """

        super().__init__()
        total_pad = (dilation * (kernel_size - 1))
        begin = total_pad // 2
        end = total_pad - begin
        # remove padding='same' here, cause onnx don't support dilation + 'same' padding
        blocks = [
            nn.Sequential(
                nn.Conv1D(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    # make sure output T == input T
                    padding=((0, 0), (0, 0), (begin, end))),
                nn.ReLU(),
                nn.BatchNorm1D(channels), ) for _ in range(n)
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            x(Tensor): Batch of input sequences (B, hidden_size, Tmax).
        Returns:
            Tensor: The residual output (B, hidden_size, Tmax).
        """
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

    def forward(self, text: paddle.Tensor, tone: paddle.Tensor=None):
        """Calculate forward propagation.
        Args:
            text(Tensor(int64)): Batch of padded token ids (B, Tmax).
            tones(Tensor, optional(int64)): Batch of padded tone ids (B, Tmax).
        Returns:
            Tensor: The residual output (B, Tmax, embedding_size).
        """

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
    """SpeedySpeech encoder module.
    Args:
        vocab_size (int): Dimension of the inputs.
        tone_size (Optional[int]): Number of tones.
        hidden_size (int): Number of encoder hidden units.
        kernel_size (int): Kernel size of encoder.
        dilations (List[int]): Dilations of encoder.
        spk_num (Optional[int]): Number of speakers. 
    """

    def __init__(self,
                 vocab_size: int,
                 tone_size: int,
                 hidden_size: int=128,
                 kernel_size: int=3,
                 dilations: List[int]=[1, 3, 9, 27, 1, 3, 9, 27, 1, 1],
                 spk_num=None):

        super().__init__()
        self.embedding = TextEmbedding(
            vocab_size,
            hidden_size,
            tone_size,
            padding_idx=0,
            tone_padding_idx=0)

        if spk_num:
            self.spk_emb = nn.Embedding(
                num_embeddings=spk_num,
                embedding_dim=hidden_size,
                padding_idx=0)
        else:
            self.spk_emb = None

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
            nn.BatchNorm1D(hidden_size), )
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self,
                text: paddle.Tensor,
                tones: paddle.Tensor,
                spk_id: paddle.Tensor=None):
        """Encoder input sequence.
        Args:
            text(Tensor(int64)): Batch of padded token ids (B, Tmax).
            tones(Tensor, optional(int64)): Batch of padded tone ids (B, Tmax).
            spk_id(Tnesor, optional(int64)): Batch of speaker ids (B,)

        Returns:
            Tensor: Output tensor (B, Tmax, hidden_size).
        """
        embedding = self.embedding(text, tones)
        if self.spk_emb:
            embedding += self.spk_emb(spk_id).unsqueeze(1)
        embedding = self.prenet(embedding)
        x = self.res_blocks(embedding.transpose([0, 2, 1])).transpose([0, 2, 1])
        # (B, T, dim)
        x = embedding + self.postnet1(x)
        x = self.postnet2(x.transpose([0, 2, 1])).transpose([0, 2, 1])
        x = self.linear(x)
        return x


class DurationPredictor(nn.Layer):
    def __init__(self, hidden_size: int=128):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualBlock(hidden_size, 4, 1, n=1),
            ResidualBlock(hidden_size, 3, 1, n=1),
            ResidualBlock(hidden_size, 1, 1, n=1), )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            x(Tensor): Batch of input sequences (B, Tmax, hidden_size).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        x = self.layers(x.transpose([0, 2, 1])).transpose([0, 2, 1])
        x = self.linear(x)
        return paddle.squeeze(x, -1)


class SpeedySpeechDecoder(nn.Layer):
    def __init__(self,
                 hidden_size: int=128,
                 output_size: int=80,
                 kernel_size: int=3,
                 dilations: List[int]=[
                     1, 3, 9, 27, 1, 3, 9, 27, 1, 3, 9, 27, 1, 3, 9, 27, 1, 1
                 ]):
        """SpeedySpeech decoder module.
        Args:
            hidden_size (int): Number of decoder hidden units.
            kernel_size (int): Kernel size of decoder.
            output_size (int): Dimension of the outputs.
            dilations (List[int]): Dilations of decoder.
        """
        super().__init__()
        res_blocks = [
            ResidualBlock(hidden_size, kernel_size, d, n=2) for d in dilations
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        self.postnet1 = nn.Sequential(nn.Linear(hidden_size, hidden_size))
        self.postnet2 = ResidualBlock(hidden_size, kernel_size, 1, n=2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Decoder input sequence.
        Args:
            x(Tensor): Input tensor (B, time, hidden_size).

        Returns:
            Tensor: Output tensor (B, time, output_size).
        """
        xx = self.res_blocks(x.transpose([0, 2, 1])).transpose([0, 2, 1])
        x = x + self.postnet1(xx)
        x = self.postnet2(x.transpose([0, 2, 1])).transpose([0, 2, 1])
        x = self.linear(x)
        return x


class SpeedySpeech(nn.Layer):
    def __init__(
            self,
            vocab_size,
            encoder_hidden_size: int=128,
            encoder_kernel_size: int=3,
            encoder_dilations: List[int]=[1, 3, 9, 27, 1, 3, 9, 27, 1, 1],
            duration_predictor_hidden_size: int=128,
            decoder_hidden_size: int=128,
            decoder_output_size: int=80,
            decoder_kernel_size: int=3,
            decoder_dilations: List[
                int]=[1, 3, 9, 27, 1, 3, 9, 27, 1, 3, 9, 27, 1, 3, 9, 27, 1, 1],
            tone_size: int=None,
            spk_num: int=None,
            init_type: str="xavier_uniform",
            positional_dropout_rate: int=0.1):
        """Initialize SpeedySpeech module.
        Args:
            vocab_size (int): Dimension of the inputs.
            encoder_hidden_size (int): Number of encoder hidden units.
            encoder_kernel_size (int): Kernel size of encoder.
            encoder_dilations (List[int]): Dilations of encoder.
            duration_predictor_hidden_size (int): Number of duration predictor hidden units.
            decoder_hidden_size (int): Number of decoder hidden units.
            decoder_kernel_size (int): Kernel size of decoder.
            decoder_dilations (List[int]): Dilations of decoder.
            decoder_output_size (int): Dimension of the outputs.
            tone_size (Optional[int]): Number of tones.
            spk_num (Optional[int]): Number of speakers. 
            init_type (str): How to initialize transformer parameters.
    
        """
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        encoder = SpeedySpeechEncoder(vocab_size, tone_size,
                                      encoder_hidden_size, encoder_kernel_size,
                                      encoder_dilations, spk_num)
        duration_predictor = DurationPredictor(duration_predictor_hidden_size)
        decoder = SpeedySpeechDecoder(decoder_hidden_size, decoder_output_size,
                                      decoder_kernel_size, decoder_dilations)
        self.position_enc = ScaledPositionalEncoding(encoder_hidden_size,
                                                     positional_dropout_rate)

        self.encoder = encoder
        self.duration_predictor = duration_predictor
        self.decoder = decoder
        # define length regulator
        self.length_regulator = LengthRegulator()

        nn.initializer.set_global_initializer(None)

    def forward(self,
                text: paddle.Tensor,
                tones: paddle.Tensor,
                durations: paddle.Tensor,
                spk_id: paddle.Tensor=None):
        """Calculate forward propagation.
        Args:
            text(Tensor(int64)): Batch of padded token ids (B, Tmax).
            durations(Tensor(int64)): Batch of padded durations (B, Tmax).
            tones(Tensor, optional(int64)): Batch of padded tone ids  (B, Tmax).
            spk_id(Tnesor, optional(int64)): Batch of speaker ids (B,)

        Returns:
            Tensor: Output tensor (B, T_frames, decoder_output_size).
            Tensor: Predicted durations (B, Tmax).
        """
        # input of embedding must be int64
        text = paddle.cast(text, 'int64')
        tones = paddle.cast(tones, 'int64')
        if spk_id is not None:
            spk_id = paddle.cast(spk_id, 'int64')
        durations = paddle.cast(durations, 'int64')
        encodings = self.encoder(text, tones, spk_id)
        pred_durations = self.duration_predictor(encodings.detach())
        # expand encodings
        durations_to_expand = durations
        encodings = self.length_regulator(encodings, durations_to_expand)
        encodings = self.position_enc(encodings)
        # decode
        decoded = self.decoder(encodings)
        return decoded, pred_durations

    def inference(self,
                  text: paddle.Tensor,
                  tones: paddle.Tensor=None,
                  durations: paddle.Tensor=None,
                  spk_id: paddle.Tensor=None):
        """Generate the sequence of features given the sequences of characters.
        Args:
            text(Tensor(int64)): Input sequence of characters (T,).
            tones(Tensor, optional(int64)): Batch of padded tone ids (T, ).
            durations(Tensor, optional (int64)): Groundtruth of duration (T,).
            spk_id(Tensor, optional(int64), optional): spk ids (1,). (Default value = None)

        Returns:
            Tensor: logmel (T, decoder_output_size).
        """
        # input of embedding must be int64
        text = paddle.cast(text, 'int64')
        text = text.unsqueeze(0)
        if tones is not None:
            tones = paddle.cast(tones, 'int64')
            tones = tones.unsqueeze(0)

        encodings = self.encoder(text, tones, spk_id)

        if durations is None:
            # (1, T)
            pred_durations = self.duration_predictor(encodings)
            durations_to_expand = paddle.round(pred_durations.exp())
            durations_to_expand = durations_to_expand.astype(paddle.int64)
        else:
            durations_to_expand = durations
        encodings = self.length_regulator(
            encodings, durations_to_expand, is_inference=True)
        encodings = self.position_enc(encodings)
        decoded = self.decoder(encodings)
        return decoded[0]


class SpeedySpeechInference(nn.Layer):
    def __init__(self, normalizer, speedyspeech_model):
        super().__init__()
        self.normalizer = normalizer
        self.acoustic_model = speedyspeech_model

    def forward(self, phones, tones, spk_id=None, durations=None):
        normalized_mel = self.acoustic_model.inference(
            phones, tones, durations=durations, spk_id=spk_id)
        logmel = self.normalizer.inverse(normalized_mel)
        return logmel
