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
"""Stochastic duration predictor modules in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""
import math
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.t2s.models.vits.flow import ConvFlow
from paddlespeech.t2s.models.vits.flow import DilatedDepthSeparableConv
from paddlespeech.t2s.models.vits.flow import ElementwiseAffineFlow
from paddlespeech.t2s.models.vits.flow import FlipFlow
from paddlespeech.t2s.models.vits.flow import LogFlow


class StochasticDurationPredictor(nn.Layer):
    """Stochastic duration predictor module.
    This is a module of stochastic duration predictor described in `Conditional
    Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech`_.
    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2106.06103
    """

    def __init__(
            self,
            channels: int=192,
            kernel_size: int=3,
            dropout_rate: float=0.5,
            flows: int=4,
            dds_conv_layers: int=3,
            global_channels: int=-1, ):
        """Initialize StochasticDurationPredictor module.
        Args:
            channels (int):
                Number of channels.
            kernel_size (int):
                Kernel size.
            dropout_rate (float):
                Dropout rate.
            flows (int):
                Number of flows.
            dds_conv_layers (int):
                Number of conv layers in DDS conv.
            global_channels (int):
                Number of global conditioning channels.
        """
        super().__init__()

        self.pre = nn.Conv1D(channels, channels, 1)
        self.dds = DilatedDepthSeparableConv(
            channels,
            kernel_size,
            layers=dds_conv_layers,
            dropout_rate=dropout_rate, )
        self.proj = nn.Conv1D(channels, channels, 1)

        self.log_flow = LogFlow()
        self.flows = nn.LayerList()
        self.flows.append(ElementwiseAffineFlow(2))
        for i in range(flows):
            self.flows.append(
                ConvFlow(
                    2,
                    channels,
                    kernel_size,
                    layers=dds_conv_layers, ))
            self.flows.append(FlipFlow())

        self.post_pre = nn.Conv1D(1, channels, 1)
        self.post_dds = DilatedDepthSeparableConv(
            channels,
            kernel_size,
            layers=dds_conv_layers,
            dropout_rate=dropout_rate, )
        self.post_proj = nn.Conv1D(channels, channels, 1)
        self.post_flows = nn.LayerList()
        self.post_flows.append(ElementwiseAffineFlow(2))
        for i in range(flows):
            self.post_flows.append(
                ConvFlow(
                    2,
                    channels,
                    kernel_size,
                    layers=dds_conv_layers, ))
            self.post_flows.append(FlipFlow())

        if global_channels > 0:
            self.global_conv = nn.Conv1D(global_channels, channels, 1)

    def forward(
            self,
            x: paddle.Tensor,
            x_mask: paddle.Tensor,
            w: Optional[paddle.Tensor]=None,
            g: Optional[paddle.Tensor]=None,
            inverse: bool=False,
            noise_scale: float=1.0, ) -> paddle.Tensor:
        """Calculate forward propagation.
        Args:
            x (Tensor):
                Input tensor (B, channels, T_text).
            x_mask (Tensor):
                Mask tensor (B, 1, T_text).
            w (Optional[Tensor]):
                Duration tensor (B, 1, T_text).
            g (Optional[Tensor]):
                Global conditioning tensor (B, channels, 1)
            inverse (bool):
                Whether to inverse the flow.
            noise_scale (float):
                Noise scale value.
        Returns:
            Tensor: 
                If not inverse, negative log-likelihood (NLL) tensor (B,).
                If inverse, log-duration tensor (B, 1, T_text).
        """
        # stop gradient
        # x = x.detach()  
        x = self.pre(x)
        if g is not None:
            # stop gradient
            x = x + self.global_conv(g.detach())
        x = self.dds(x, x_mask)
        x = self.proj(x) * x_mask

        if not inverse:
            assert w is not None, "w must be provided."
            h_w = self.post_pre(w)
            h_w = self.post_dds(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (paddle.randn([paddle.shape(w)[0], 2, paddle.shape(w)[2]]) *
                   x_mask)
            z_q = e_q
            logdet_tot_q = 0.0
            for i, flow in enumerate(self.post_flows):
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = paddle.split(z_q, [1, 1], 1)
            u = F.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            tmp1 = (F.log_sigmoid(z_u) + F.log_sigmoid(-z_u)) * x_mask
            logdet_tot_q += paddle.sum(tmp1, [1, 2])
            tmp2 = -0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask
            logq = (paddle.sum(tmp2, [1, 2]) - logdet_tot_q)
            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = paddle.concat([z0, z1], 1)
            for flow in self.flows:
                z, logdet = flow(z, x_mask, g=x, inverse=inverse)
                logdet_tot = logdet_tot + logdet
            tmp3 = 0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask
            nll = (paddle.sum(tmp3, [1, 2]) - logdet_tot)
            # (B,)
            return nll + logq
        else:
            flows = list(reversed(self.flows))
            # remove a useless vflow
            flows = flows[:-2] + [flows[-1]]
            z = (paddle.randn([paddle.shape(x)[0], 2, paddle.shape(x)[2]]) *
                 noise_scale)
            for flow in flows:
                z = flow(z, x_mask, g=x, inverse=inverse)
            z0, z1 = paddle.split(z, 2, axis=1)
            logw = z0
            return logw
