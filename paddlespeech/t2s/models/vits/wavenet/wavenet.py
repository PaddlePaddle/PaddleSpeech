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
# Modified from espnet(https://github.com/espnet/espnet)
import math
from typing import Optional

import paddle
from paddle import nn

from paddlespeech.t2s.models.vits.wavenet.residual_block import ResidualBlock


class WaveNet(nn.Layer):
    """WaveNet with global conditioning."""
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        layers: int = 30,
        stacks: int = 3,
        base_dilation: int = 2,
        residual_channels: int = 64,
        aux_channels: int = -1,
        gate_channels: int = 128,
        skip_channels: int = 64,
        global_channels: int = -1,
        dropout_rate: float = 0.0,
        bias: bool = True,
        use_weight_norm: bool = True,
        use_first_conv: bool = False,
        use_last_conv: bool = False,
        scale_residual: bool = False,
        scale_skip_connect: bool = False,
    ):
        """Initialize WaveNet module.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            kernel_size (int):
                Kernel size of dilated convolution.
            layers (int):
                Number of residual block layers.
            stacks (int):
                Number of stacks i.e., dilation cycles.
            base_dilation (int):
                Base dilation factor.
            residual_channels (int):
                Number of channels in residual conv.
            gate_channels (int):
                Number of channels in gated conv.
            skip_channels (int):
                Number of channels in skip conv.
            aux_channels (int):
                Number of channels for local conditioning feature.
            global_channels (int):
                Number of channels for global conditioning feature.
            dropout_rate (float):
                Dropout rate. 0.0 means no dropout applied.
            bias (bool):
                Whether to use bias parameter in conv layer.
            use_weight_norm (bool):
                Whether to use weight norm. If set to true, it will be applied to all of the conv layers.
            use_first_conv (bool):
                Whether to use the first conv layers.
            use_last_conv (bool):
                Whether to use the last conv layers.
            scale_residual (bool):
                Whether to scale the residual outputs.
            scale_skip_connect (bool):
                Whether to scale the skip connection outputs.

        """
        super().__init__()
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.base_dilation = base_dilation
        self.use_first_conv = use_first_conv
        self.use_last_conv = use_last_conv
        self.scale_skip_connect = scale_skip_connect

        # check the number of layers and stacks
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        # define first convolution
        if self.use_first_conv:
            self.first_conv = nn.Conv1D(in_channels,
                                        residual_channels,
                                        kernel_size=1,
                                        bias_attr=True)

        # define residual blocks
        self.conv_layers = nn.LayerList()
        for layer in range(layers):
            dilation = base_dilation**(layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                global_channels=global_channels,
                dilation=dilation,
                dropout_rate=dropout_rate,
                bias=bias,
                scale_residual=scale_residual,
            )
            self.conv_layers.append(conv)

        # define output layers
        if self.use_last_conv:
            self.last_conv = nn.Sequential(
                nn.ReLU(),
                nn.Conv1D(skip_channels,
                          skip_channels,
                          kernel_size=1,
                          bias_attr=True),
                nn.ReLU(),
                nn.Conv1D(skip_channels,
                          out_channels,
                          kernel_size=1,
                          bias_attr=True),
            )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(
        self,
        x: paddle.Tensor,
        x_mask: Optional[paddle.Tensor] = None,
        c: Optional[paddle.Tensor] = None,
        g: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor):
                Input noise signal (B, 1, T) if use_first_conv else (B, residual_channels, T).
            x_mask (Optional[Tensor]):
                Mask tensor (B, 1, T).
            c (Optional[Tensor]):
                Local conditioning features (B, aux_channels, T).
            g (Optional[Tensor]):
                Global conditioning features (B, global_channels, 1).

        Returns:
            Tensor:
                Output tensor (B, out_channels, T) if use_last_conv else(B, residual_channels, T).

        """
        # encode to hidden representation
        if self.use_first_conv:
            x = self.first_conv(x)

        # residual block
        skips = 0.0
        for f in self.conv_layers:
            x, h = f(x, x_mask=x_mask, c=c, g=g)
            skips = skips + h
        x = skips
        if self.scale_skip_connect:
            x = x * math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        if self.use_last_conv:
            x = self.last_conv(x)

        return x

    def apply_weight_norm(self):
        def _apply_weight_norm(layer):
            if isinstance(layer, (nn.Conv1D, nn.Conv2D)):
                nn.utils.weight_norm(layer)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        def _remove_weight_norm(layer):
            try:
                nn.utils.remove_weight_norm(layer)
            except ValueError:
                pass

        self.apply(_remove_weight_norm)
