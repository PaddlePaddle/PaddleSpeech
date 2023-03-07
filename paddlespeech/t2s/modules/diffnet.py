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

from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.utils.initialize import _calculate_fan_in_and_fan_out
from paddlespeech.utils.initialize import kaiming_normal_
from paddlespeech.utils.initialize import kaiming_uniform_
from paddlespeech.utils.initialize import uniform_
from paddlespeech.utils.initialize import zeros_


def Conv1D(*args, **kwargs):
    layer = nn.Conv1D(*args, **kwargs)
    # Initialize the weight to be consistent with the official
    kaiming_normal_(layer.weight)

    # Initialization is consistent with torch
    if layer.bias is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(layer.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            uniform_(layer.bias, -bound, bound)
    return layer


# Initialization is consistent with torch
def Linear(*args, **kwargs):
    layer = nn.Linear(*args, **kwargs)
    kaiming_uniform_(layer.weight, a=math.sqrt(5))
    if layer.bias is not None:
        fan_in, _ = _calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        uniform_(layer.bias, -bound, bound)
    return layer


class ResidualBlock(nn.Layer):
    """ResidualBlock
    """

    def __init__(self, encoder_hidden, residual_channels, gate_channels,
                 kernel_size, dilation):
        super().__init__()
        self.dilated_conv = Conv1D(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=dilation,
            dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1D(encoder_hidden, gate_channels, 1)
        self.output_projection = Conv1D(residual_channels, gate_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        """_summary_

        Args:
            nn (_type_): _description_
        """
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        gate, filter = paddle.chunk(y, 2, axis=1)
        y = F.sigmoid(gate) * paddle.tanh(filter)

        y = self.output_projection(y)
        residual, skip = paddle.chunk(y, 2, axis=1)
        return (x + residual) / math.sqrt(2.0), skip


class SinusoidalPosEmb(nn.Layer):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """_summary_

        Args:
            nn (_type_): _description_
        """
        x = paddle.cast(x, 'float32')
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(paddle.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = paddle.concat([emb.sin(), emb.cos()], axis=-1)
        return emb


class DiffNet(nn.Layer):
    def __init__(
            self,
            in_channels: int=80,
            out_channels: int=80,
            kernel_size: int=3,
            layers: int=20,
            stacks: int=5,
            residual_channels: int=256,
            gate_channels: int=512,
            skip_channels: int=256,
            aux_channels: int=256,
            dropout: float=0.,
            bias: bool=True,
            use_weight_norm: bool=False,
            init_type: str="kaiming_normal", ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.aux_channels = aux_channels
        self.residual_channels = residual_channels
        self.gate_channels = gate_channels
        self.kernel_size = kernel_size
        self.dilation_cycle_length = layers // stacks
        self.skip_channels = skip_channels

        self.input_projection = Conv1D(self.in_channels, self.residual_channels,
                                       1)
        self.diffusion_embedding = SinusoidalPosEmb(self.residual_channels)
        dim = self.residual_channels
        self.mlp = nn.Sequential(
            Linear(dim, dim * 4), nn.Mish(), Linear(dim * 4, dim))
        self.residual_layers = nn.LayerList([
            ResidualBlock(
                encoder_hidden=self.aux_channels,
                residual_channels=self.residual_channels,
                gate_channels=self.gate_channels,
                kernel_size=self.kernel_size,
                dilation=2**(i % self.dilation_cycle_length))
            for i in range(self.layers)
        ])
        self.skip_projection = Conv1D(self.residual_channels,
                                      self.skip_channels, 1)
        self.output_projection = Conv1D(self.residual_channels,
                                        self.out_channels, 1)
        zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """

        :param spec: [B, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec
        x = self.input_projection(x)  # x [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)
        x = paddle.sum(
            paddle.stack(skip), axis=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]
        return x
