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
"""Variance predictor related modules."""
import paddle
from paddle import nn
from typeguard import check_argument_types

from paddlespeech.t2s.modules.layer_norm import LayerNorm
from paddlespeech.t2s.modules.masked_fill import masked_fill


class VariancePredictor(nn.Layer):
    """Variance predictor module.

    This is a module of variacne predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    """
    def __init__(
        self,
        idim: int,
        n_layers: int = 2,
        n_chans: int = 384,
        kernel_size: int = 3,
        bias: bool = True,
        dropout_rate: float = 0.5,
    ):
        """Initilize duration predictor module.

        Args:
            idim (int): 
                Input dimension.
            n_layers (int, optional): 
                Number of convolutional layers.
            n_chans (int, optional): 
                Number of channels of convolutional layers.
            kernel_size (int, optional): 
                Kernel size of convolutional layers.
            dropout_rate (float, optional): 
                Dropout rate.
        """
        assert check_argument_types()
        super().__init__()
        self.conv = nn.LayerList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv.append(
                nn.Sequential(
                    nn.Conv1D(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias_attr=True,
                    ),
                    nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    nn.Dropout(dropout_rate),
                ))

        self.linear = nn.Linear(n_chans, 1, bias_attr=True)

    def forward(self,
                xs: paddle.Tensor,
                x_masks: paddle.Tensor = None) -> paddle.Tensor:
        """Calculate forward propagation.

        Args:
            xs (Tensor): 
                Batch of input sequences (B, Tmax, idim).
            x_masks (Tensor(bool), optional): 
                Batch of masks indicating padded part (B, Tmax, 1).

        Returns:
            Tensor: 
                Batch of predicted sequences (B, Tmax, 1).
        """
        # (B, idim, Tmax)
        xs = xs.transpose([0, 2, 1])
        # (B, C, Tmax)
        for f in self.conv:
            # (B, C, Tmax)
            xs = f(xs)
        # (B, Tmax, 1)
        xs = self.linear(xs.transpose([0, 2, 1]))

        if x_masks is not None:
            xs = masked_fill(xs, x_masks, 0.0)
        return xs
