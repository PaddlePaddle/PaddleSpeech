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
"""Residual affine coupling modules in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
from paddle import nn

from paddlespeech.t2s.models.vits.flow import FlipFlow
from paddlespeech.t2s.models.vits.wavenet.wavenet import WaveNet


class ResidualAffineCouplingBlock(nn.Layer):
    """Residual affine coupling block module.

    This is a module of residual affine coupling block, which used as "Flow" in
    `Conditional Variational Autoencoder with Adversarial Learning for End-to-End
    Text-to-Speech`_.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    """
    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 192,
        flows: int = 4,
        kernel_size: int = 5,
        base_dilation: int = 1,
        layers: int = 4,
        global_channels: int = -1,
        dropout_rate: float = 0.0,
        use_weight_norm: bool = True,
        bias: bool = True,
        use_only_mean: bool = True,
    ):
        """Initilize ResidualAffineCouplingBlock module.

        Args:
            in_channels (int):
                Number of input channels.
            hidden_channels (int):
                Number of hidden channels.
            flows (int):
                Number of flows.
            kernel_size (int):
                Kernel size for WaveNet.
            base_dilation (int):
                Base dilation factor for WaveNet.
            layers (int):
                Number of layers of WaveNet.
            stacks (int):
                Number of stacks of WaveNet.
            global_channels (int):
                Number of global channels.
            dropout_rate (float):
                Dropout rate.
            use_weight_norm (bool):
                Whether to use weight normalization in WaveNet.
            bias (bool):
                Whether to use bias paramters in WaveNet.
            use_only_mean (bool):
                Whether to estimate only mean.

        """
        super().__init__()

        self.flows = nn.LayerList()
        for i in range(flows):
            self.flows.append(
                ResidualAffineCouplingLayer(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    base_dilation=base_dilation,
                    layers=layers,
                    stacks=1,
                    global_channels=global_channels,
                    dropout_rate=dropout_rate,
                    use_weight_norm=use_weight_norm,
                    bias=bias,
                    use_only_mean=use_only_mean,
                ))
            self.flows.append(FlipFlow())

    def forward(
        self,
        x: paddle.Tensor,
        x_mask: paddle.Tensor,
        g: Optional[paddle.Tensor] = None,
        inverse: bool = False,
    ) -> paddle.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor):
                Input tensor (B, in_channels, T).
            x_mask (Tensor):
                Length tensor (B, 1, T).
            g (Optional[Tensor]):
                Global conditioning tensor (B, global_channels, 1).
            inverse (bool):
                Whether to inverse the flow.

        Returns:
            Tensor: Output tensor (B, in_channels, T).

        """
        if not inverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, inverse=inverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, inverse=inverse)
        return x


class ResidualAffineCouplingLayer(nn.Layer):
    """Residual affine coupling layer."""
    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        base_dilation: int = 1,
        layers: int = 5,
        stacks: int = 1,
        global_channels: int = -1,
        dropout_rate: float = 0.0,
        use_weight_norm: bool = True,
        bias: bool = True,
        use_only_mean: bool = True,
    ):
        """Initialzie ResidualAffineCouplingLayer module.

        Args:
            in_channels (int):
                Number of input channels.
            hidden_channels (int):
                Number of hidden channels.
            kernel_size (int):
                Kernel size for WaveNet.
            base_dilation (int):
                Base dilation factor for WaveNet.
            layers (int):
                Number of layers of WaveNet.
            stacks (int):
                Number of stacks of WaveNet.
            global_channels (int):
                Number of global channels.
            dropout_rate (float):
                Dropout rate.
            use_weight_norm (bool):
                Whether to use weight normalization in WaveNet.
            bias (bool):
                Whether to use bias paramters in WaveNet.
            use_only_mean (bool):
                Whether to estimate only mean.

        """
        assert in_channels % 2 == 0, "in_channels should be divisible by 2"
        super().__init__()
        self.half_channels = in_channels // 2
        self.use_only_mean = use_only_mean

        # define modules
        self.input_conv = nn.Conv1D(
            self.half_channels,
            hidden_channels,
            1,
        )
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
            scale_skip_connect=True,
        )
        if use_only_mean:
            self.proj = nn.Conv1D(
                hidden_channels,
                self.half_channels,
                1,
            )
        else:
            self.proj = nn.Conv1D(
                hidden_channels,
                self.half_channels * 2,
                1,
            )

        weight = paddle.zeros(paddle.shape(self.proj.weight))

        self.proj.weight = paddle.create_parameter(
            shape=weight.shape,
            dtype=str(weight.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(weight))

        bias = paddle.zeros(paddle.shape(self.proj.bias))

        self.proj.bias = paddle.create_parameter(
            shape=bias.shape,
            dtype=str(bias.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(bias))

    def forward(
        self,
        x: paddle.Tensor,
        x_mask: paddle.Tensor,
        g: Optional[paddle.Tensor] = None,
        inverse: bool = False,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor):
                Input tensor (B, in_channels, T).
            x_lengths (Tensor):
                Length tensor (B,).
            g (Optional[Tensor]):
                Global conditioning tensor (B, global_channels, 1).
            inverse (bool):
                Whether to inverse the flow.

        Returns:
            Tensor:
                Output tensor (B, in_channels, T).
            Tensor:
                Log-determinant tensor for NLL (B,) if not inverse.

        """
        xa, xb = paddle.split(x, 2, axis=1)
        h = self.input_conv(xa) * x_mask
        h = self.encoder(h, x_mask, g=g)
        stats = self.proj(h) * x_mask
        if not self.use_only_mean:
            m, logs = paddle.split(stats, 2, axis=1)
        else:
            m = stats
            logs = paddle.zeros(paddle.shape(m))

        if not inverse:
            xb = m + xb * paddle.exp(logs) * x_mask
            x = paddle.concat([xa, xb], 1)
            logdet = paddle.sum(logs, [1, 2])
            return x, logdet
        else:
            xb = (xb - m) * paddle.exp(-logs) * x_mask
            x = paddle.concat([xa, xb], 1)
            return x
