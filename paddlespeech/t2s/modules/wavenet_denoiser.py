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
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import ppdiffusers
from paddle import nn
from ppdiffusers.models.embeddings import Timesteps
from ppdiffusers.schedulers import DDPMScheduler

from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.residual_block import WaveNetResidualBlock


class WaveNetDenoiser(nn.Layer):
    """A Mel-Spectrogram Denoiser modified from WaveNet

    Args:
        in_channels (int, optional): 
            Number of channels of the input mel-spectrogram, by default 80
        out_channels (int, optional): 
            Number of channels of the output mel-spectrogram, by default 80
        kernel_size (int, optional): 
            Kernel size of the residual blocks inside, by default 3
        layers (int, optional): 
            Number of residual blocks inside, by default 20
        stacks (int, optional):
            The number of groups to split the residual blocks into, by default 5
            Within each group, the dilation of the residual block grows exponentially.
        residual_channels (int, optional): 
            Residual channel of the residual blocks, by default 256
        gate_channels (int, optional): 
            Gate channel of the residual blocks, by default 512
        skip_channels (int, optional): 
            Skip channel of the residual blocks, by default 256
        aux_channels (int, optional): 
            Auxiliary channel of the residual blocks, by default 256
        dropout (float, optional): 
            Dropout of the residual blocks, by default 0.
        bias (bool, optional): 
            Whether to use bias in residual blocks, by default True
        use_weight_norm (bool, optional): 
            Whether to use weight norm in all convolutions, by default False
    """

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

        # initialize parameters
        initialize(self, init_type)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size

        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        self.first_t_emb = nn.Sequential(
            Timesteps(
                residual_channels,
                flip_sin_to_cos=False,
                downscale_freq_shift=1),
            nn.Linear(residual_channels, residual_channels * 4),
            nn.Mish(), nn.Linear(residual_channels * 4, residual_channels))
        self.t_emb_layers = nn.LayerList([
            nn.Linear(residual_channels, residual_channels)
            for _ in range(layers)
        ])

        self.first_conv = nn.Conv1D(
            in_channels, residual_channels, 1, bias_attr=True)
        self.first_act = nn.ReLU()

        self.conv_layers = nn.LayerList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            conv = WaveNetResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                bias=bias)
            self.conv_layers.append(conv)

        final_conv = nn.Conv1D(skip_channels, out_channels, 1, bias_attr=True)
        nn.initializer.Constant(0.0)(final_conv.weight)
        self.last_conv_layers = nn.Sequential(nn.ReLU(),
                                              nn.Conv1D(
                                                  skip_channels,
                                                  skip_channels,
                                                  1,
                                                  bias_attr=True),
                                              nn.ReLU(), final_conv)

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, t, c):
        """Denoise mel-spectrogram.

        Args:
            x(Tensor): 
                Shape (N, C_in, T), The input mel-spectrogram.
            t(Tensor): 
                Shape (N), The timestep input.
            c(Tensor): 
                Shape (N, C_aux, T'). The auxiliary input (e.g. fastspeech2 encoder output). 

        Returns:
            Tensor: Shape (N, C_out, T), the denoised mel-spectrogram.
        """
        assert c.shape[-1] == x.shape[-1]

        if t.shape[0] != x.shape[0]:
            t = t.tile([x.shape[0]])
        t_emb = self.first_t_emb(t)
        t_embs = [
            t_emb_layer(t_emb)[..., None] for t_emb_layer in self.t_emb_layers
        ]

        x = self.first_conv(x)
        x = self.first_act(x)
        skips = 0
        for f, t in zip(self.conv_layers, t_embs):
            x = x + t
            x, s = f(x, c)
            skips += s
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        x = self.last_conv_layers(skips)
        return x

    def apply_weight_norm(self):
        """Recursively apply weight normalization to all the Convolution layers
        in the sublayers.
        """

        def _apply_weight_norm(layer):
            if isinstance(layer, (nn.Conv1D, nn.Conv2D)):
                nn.utils.weight_norm(layer)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Recursively remove weight normalization from all the Convolution 
        layers in the sublayers.
        """

        def _remove_weight_norm(layer):
            try:
                nn.utils.remove_weight_norm(layer)
            except ValueError:
                pass

        self.apply(_remove_weight_norm)

