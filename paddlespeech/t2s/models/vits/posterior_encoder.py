# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""Text encoder module in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""
from typing import Optional
from typing import Tuple

import paddle
from paddle import nn

from paddlespeech.t2s.models.vits.wavenet.wavenet import WaveNet
from paddlespeech.t2s.modules.nets_utils import make_non_pad_mask


class PosteriorEncoder(nn.Layer):
    """Posterior encoder module in VITS.

    This is a module of posterior encoder described in `Conditional Variational
    Autoencoder with Adversarial Learning for End-to-End Text-to-Speech`_.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558
    """

    def __init__(
            self,
            in_channels: int=513,
            out_channels: int=192,
            hidden_channels: int=192,
            kernel_size: int=5,
            layers: int=16,
            stacks: int=1,
            base_dilation: int=1,
            global_channels: int=-1,
            dropout_rate: float=0.0,
            bias: bool=True,
            use_weight_norm: bool=True, ):
        """Initilialize PosteriorEncoder module.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            hidden_channels (int):
                Number of hidden channels.
            kernel_size (int):
                Kernel size in WaveNet.
            layers (int):
                Number of layers of WaveNet.
            stacks (int):
                Number of repeat stacking of WaveNet.
            base_dilation (int):
                Base dilation factor.
            global_channels (int):
                Number of global conditioning channels.
            dropout_rate (float):
                Dropout rate.
            bias (bool):
                Whether to use bias parameters in conv.
            use_weight_norm (bool):
                Whether to apply weight norm.

        """
        super().__init__()

        # define modules
        self.input_conv = nn.Conv1D(in_channels, hidden_channels, 1)
        self.encoder = WaveNet(
            in_channels=-1,
            out_channels=-1,
            kernel_size=kernel_size,
            layers=layers,
            stacks=stacks,
            base_dilation=base_dilation,
            residual_channels=hidden_channels,
            aux_channels=-1,
            gate_channels=hidden_channels * 2,
            skip_channels=hidden_channels,
            global_channels=global_channels,
            dropout_rate=dropout_rate,
            bias=bias,
            use_weight_norm=use_weight_norm,
            use_first_conv=False,
            use_last_conv=False,
            scale_residual=False,
            scale_skip_connect=True, )
        self.proj = nn.Conv1D(hidden_channels, out_channels * 2, 1)

    def forward(
            self,
            x: paddle.Tensor,
            x_lengths: paddle.Tensor,
            g: Optional[paddle.Tensor]=None
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Calculate forward propagation.

        Args:
            x (Tensor):
                Input tensor (B, in_channels, T_feats).
            x_lengths (Tensor):
                Length tensor (B,).
            g (Optional[Tensor]):
                Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor:
                Encoded hidden representation tensor (B, out_channels, T_feats).
            Tensor:
                Projected mean tensor (B, out_channels, T_feats).
            Tensor:
                Projected scale tensor (B, out_channels, T_feats).
            Tensor:
                Mask tensor for input tensor (B, 1, T_feats).

        """
        x_mask = make_non_pad_mask(x_lengths).unsqueeze(1)
        x_mask = x_mask.astype(x.dtype)
        x = self.input_conv(x) * x_mask
        x = self.encoder(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = paddle.split(stats, 2, axis=1)
        z = (m + paddle.randn(paddle.shape(m)) * paddle.exp(logs)) * x_mask

        return z, m, logs, x_mask
