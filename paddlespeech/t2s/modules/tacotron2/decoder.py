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
"""Tacotron2 decoder related modules."""
import paddle.nn.functional as F
import six
from paddle import nn


class Prenet(nn.Layer):
    """Prenet module for decoder of Spectrogram prediction network.

    This is a module of Prenet in the decoder of Spectrogram prediction network,
    which described in `Natural TTS
    Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Prenet preforms nonlinear conversion
    of inputs before input to auto-regressive lstm,
    which helps to learn diagonal attentions.

    Notes
    ----------
    This module alway applies dropout even in evaluation.
    See the detail in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(self, idim, n_layers=2, n_units=256, dropout_rate=0.5):
        """Initialize prenet module.

        Parameters
        ----------
        idim : int
            Dimension of the inputs.
        odim : int
            Dimension of the outputs.
        n_layers : int, optional
            The number of prenet layers.
        n_units : int, optional
            The number of prenet units.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.prenet = nn.LayerList()
        for layer in six.moves.range(n_layers):
            n_inputs = idim if layer == 0 else n_units
            self.prenet.append(
                nn.Sequential(nn.Linear(n_inputs, n_units), nn.ReLU()))

    def forward(self, x):
        """Calculate forward propagation.

        Parameters
        ----------
        x : Tensor
            Batch of input tensors (B, ..., idim).

        Returns
        ----------
        Tensor
            Batch of output tensors (B, ..., odim).

        """
        for i in six.moves.range(len(self.prenet)):
            # F.dropout 引入了随机, tacotron2 的 dropout 是不能去掉的
            x = F.dropout(self.prenet[i](x))
        return x


class Postnet(nn.Layer):
    """Postnet module for Spectrogram prediction network.

    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet predicts refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail sturcture of spectrogram.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
            self,
            idim,
            odim,
            n_layers=5,
            n_chans=512,
            n_filts=5,
            dropout_rate=0.5,
            use_batch_norm=True, ):
        """Initialize postnet module.

        Parameters
        ----------
        idim : int
            Dimension of the inputs.
        odim : int
            Dimension of the outputs.
        n_layers : int, optional
            The number of layers.
        n_filts : int, optional
            The number of filter size.
        n_units : int, optional
            The number of filter channels.
        use_batch_norm : bool, optional
            Whether to use batch normalization..
        dropout_rate : float, optional
            Dropout rate..
        """
        super().__init__()
        self.postnet = nn.LayerList()
        for layer in six.moves.range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            if use_batch_norm:
                self.postnet.append(
                    nn.Sequential(
                        nn.Conv1D(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias_attr=False, ),
                        nn.BatchNorm1D(ochans),
                        nn.Tanh(),
                        nn.Dropout(dropout_rate), ))
            else:
                self.postnet.append(
                    nn.Sequential(
                        nn.Conv1D(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias_attr=False, ),
                        nn.Tanh(),
                        nn.Dropout(dropout_rate), ))
        ichans = n_chans if n_layers != 1 else odim
        if use_batch_norm:
            self.postnet.append(
                nn.Sequential(
                    nn.Conv1D(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias_attr=False, ),
                    nn.BatchNorm1D(odim),
                    nn.Dropout(dropout_rate), ))
        else:
            self.postnet.append(
                nn.Sequential(
                    nn.Conv1D(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias_attr=False, ),
                    nn.Dropout(dropout_rate), ))

    def forward(self, xs):
        """Calculate forward propagation.

        Parameters
        ----------
        xs : Tensor
            Batch of the sequences of padded input tensors (B, idim, Tmax).

        Returns
        ----------
        Tensor
            Batch of padded output tensor. (B, odim, Tmax).

        """
        for i in six.moves.range(len(self.postnet)):
            xs = self.postnet[i](xs)
        return xs
