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
# Modified from espnet(https://github.com/espnet/espnet)
"""Tacotron2 encoder related modules."""
import paddle
from paddle import nn


class Encoder(nn.Layer):
    """Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in Tacotron2,
    which described in `Natural TTS Synthesis by Conditioning WaveNet on Mel
    Spectrogram Predictions`_. This is the encoder which converts either a sequence
    of characters or acoustic features into the sequence of hidden states.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
            self,
            idim,
            input_layer="embed",
            embed_dim=512,
            elayers=1,
            eunits=512,
            econv_layers=3,
            econv_chans=512,
            econv_filts=5,
            use_batch_norm=True,
            use_residual=False,
            dropout_rate=0.5,
            padding_idx=0, ):
        """Initialize Tacotron2 encoder module.
        Args:
            idim (int): 
                Dimension of the inputs.
            input_layer (str): 
                Input layer type.
            embed_dim (int, optional): 
                Dimension of character embedding.
            elayers (int, optional): 
                The number of encoder blstm layers.
            eunits (int, optional): 
                The number of encoder blstm units.
            econv_layers (int, optional): 
                The number of encoder conv layers.
            econv_filts (int, optional): 
                The number of encoder conv filter size.
            econv_chans (int, optional): 
                The number of encoder conv filter channels.
            use_batch_norm (bool, optional): 
                Whether to use batch normalization.
            use_residual (bool, optional): 
                Whether to use residual connection.
            dropout_rate (float, optional): 
                Dropout rate.

        """
        super().__init__()
        # store the hyperparameters
        self.idim = idim
        self.use_residual = use_residual

        # define network layer modules
        if input_layer == "linear":
            self.embed = nn.Linear(idim, econv_chans)
        elif input_layer == "embed":
            self.embed = nn.Embedding(idim, embed_dim, padding_idx=padding_idx)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        if econv_layers > 0:
            self.convs = nn.LayerList()
            for layer in range(econv_layers):
                ichans = (embed_dim if layer == 0 and input_layer == "embed"
                          else econv_chans)
                if use_batch_norm:
                    self.convs.append(
                        nn.Sequential(
                            nn.Conv1D(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias_attr=False, ),
                            nn.BatchNorm1D(econv_chans),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate), ))
                else:
                    self.convs += [
                        nn.Sequential(
                            nn.Conv1D(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias_attr=False, ),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate), )
                    ]
        else:
            self.convs = None
        if elayers > 0:
            iunits = econv_chans if econv_layers != 0 else embed_dim
            # batch_first=True, bidirectional=True
            self.blstm = nn.LSTM(
                iunits,
                eunits // 2,
                elayers,
                time_major=False,
                direction='bidirectional',
                bias_ih_attr=True,
                bias_hh_attr=True)
            self.blstm.flatten_parameters()
        else:
            self.blstm = None

        # # initialize
        # self.apply(encoder_init)

    def forward(self, xs, ilens=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): 
                Batch of the padded sequence. Either character ids (B, Tmax)
                or acoustic feature (B, Tmax, idim * encoder_reduction_factor). 
                Padded value should be 0.
            ilens (Tensor(int64)): 
                Batch of lengths of each input batch (B,).

        Returns:
            Tensor: 
                Batch of the sequences of encoder states(B, Tmax, eunits).
            Tensor(int64): 
                Batch of lengths of each sequence (B,)
        """
        xs = self.embed(xs).transpose([0, 2, 1])
        if self.convs is not None:
            for i in range(len(self.convs)):
                if self.use_residual:
                    xs += self.convs[i](xs)
                else:
                    xs = self.convs[i](xs)
        if self.blstm is None:
            return xs.transpose([0, 2, 1])
        if not isinstance(ilens, paddle.Tensor):
            ilens = paddle.to_tensor(ilens)
        if ilens.ndim == 0:
            ilens = ilens.unsqueeze(0)
        xs = xs.transpose([0, 2, 1])
        # for dygraph to static graph
        # self.blstm.flatten_parameters()
        # (B, Tmax, C)
        # see https://www.paddlepaddle.org.cn/documentation/docs/zh/faq/train_cn.html#paddletorch-nn-utils-rnn-pack-padded-sequencetorch-nn-utils-rnn-pad-packed-sequenceapi
        xs, _ = self.blstm(xs, sequence_length=ilens)
        hlens = ilens

        return xs, hlens

    def inference(self, x):
        """Inference.

        Args:
            x (Tensor): 
                The sequeunce of character ids (T,) or acoustic feature (T, idim * encoder_reduction_factor).

        Returns:
            Tensor: The sequences of encoder states(T, eunits).

        """
        xs = x.unsqueeze(0)
        ilens = paddle.shape(x)[0]

        return self.forward(xs, ilens)[0][0]
