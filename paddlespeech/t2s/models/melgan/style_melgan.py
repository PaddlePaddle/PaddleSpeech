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
"""StyleMelGAN Modules."""
import copy
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.t2s.models.melgan import MelGANDiscriminator as BaseDiscriminator
from paddlespeech.t2s.modules.activation import get_activation
from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.pqmf import PQMF
from paddlespeech.t2s.modules.tade_res_block import TADEResBlock


class StyleMelGANGenerator(nn.Layer):
    """Style MelGAN generator module."""
    def __init__(
        self,
        in_channels: int = 128,
        aux_channels: int = 80,
        channels: int = 64,
        out_channels: int = 1,
        kernel_size: int = 9,
        dilation: int = 2,
        bias: bool = True,
        noise_upsample_scales: List[int] = [11, 2, 2, 2],
        noise_upsample_activation: str = "leakyrelu",
        noise_upsample_activation_params: Dict[str,
                                               Any] = {"negative_slope": 0.2},
        upsample_scales: List[int] = [2, 2, 2, 2, 2, 2, 2, 2, 1],
        upsample_mode: str = "linear",
        gated_function: str = "softmax",
        use_weight_norm: bool = True,
        init_type: str = "xavier_uniform",
    ):
        """Initilize Style MelGAN generator.

        Args:
            in_channels (int): 
                Number of input noise channels.
            aux_channels (int): 
                Number of auxiliary input channels.
            channels (int): 
                Number of channels for conv layer.
            out_channels (int): 
                Number of output channels.
            kernel_size (int): 
                Kernel size of conv layers.
            dilation (int): 
                Dilation factor for conv layers.
            bias (bool): 
                Whether to add bias parameter in convolution layers.
            noise_upsample_scales (list): 
                List of noise upsampling scales.
            noise_upsample_activation (str): 
                Activation function module name for noise upsampling.
            noise_upsample_activation_params (dict): 
                Hyperparameters for the above activation function.
            upsample_scales (list): 
                List of upsampling scales.
            upsample_mode (str): 
                Upsampling mode in TADE layer.
            gated_function (str): 
                Gated function in TADEResBlock ("softmax" or "sigmoid").
            use_weight_norm (bool): 
                Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        self.in_channels = in_channels
        noise_upsample = []
        in_chs = in_channels
        for noise_upsample_scale in noise_upsample_scales:
            noise_upsample.append(
                nn.Conv1DTranspose(
                    in_chs,
                    channels,
                    noise_upsample_scale * 2,
                    stride=noise_upsample_scale,
                    padding=noise_upsample_scale // 2 +
                    noise_upsample_scale % 2,
                    output_padding=noise_upsample_scale % 2,
                    bias_attr=bias,
                ))
            noise_upsample.append(
                get_activation(noise_upsample_activation,
                               **noise_upsample_activation_params))
            in_chs = channels
        self.noise_upsample = nn.Sequential(*noise_upsample)
        self.noise_upsample_factor = np.prod(noise_upsample_scales)

        self.blocks = nn.LayerList()
        aux_chs = aux_channels
        for upsample_scale in upsample_scales:
            self.blocks.append(
                TADEResBlock(
                    in_channels=channels,
                    aux_channels=aux_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    bias=bias,
                    upsample_factor=upsample_scale,
                    upsample_mode=upsample_mode,
                    gated_function=gated_function,
                ), )
            aux_chs = channels
        self.upsample_factor = np.prod(upsample_scales)

        self.output_conv = nn.Sequential(
            nn.Conv1D(
                channels,
                out_channels,
                kernel_size,
                1,
                bias_attr=bias,
                padding=(kernel_size - 1) // 2,
            ),
            nn.Tanh(),
        )

        nn.initializer.set_global_initializer(None)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c, z=None):
        """Calculate forward propagation.

        Args:
            c (Tensor): Auxiliary input tensor (B, channels, T).
            z (Tensor): Input noise tensor (B, in_channels, 1).
        Returns:
            Tensor: Output tensor (B, out_channels, T ** prod(upsample_scales)).
        """
        # batch_max_steps(24000) == noise_upsample_factor(80) * upsample_factor(300)
        if z is None:
            z = paddle.randn([paddle.shape(c)[0], self.in_channels, 1])
        # (B, in_channels, noise_upsample_factor).
        x = self.noise_upsample(z)
        for block in self.blocks:
            x, c = block(x, c)
        x = self.output_conv(x)
        return x

    def apply_weight_norm(self):
        """Recursively apply weight normalization to all the Convolution layers
        in the sublayers.
        """
        def _apply_weight_norm(layer):
            if isinstance(layer, (nn.Conv1D, nn.Conv1DTranspose)):
                nn.utils.weight_norm(layer)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Recursively remove weight normalization from all the Convolution 
        layers in the sublayers.
        """
        def _remove_weight_norm(layer):
            try:
                if layer:
                    nn.utils.remove_weight_norm(layer)
            # add AttributeError to bypass https://github.com/PaddlePaddle/Paddle/issues/38532 temporarily
            except (ValueError, AttributeError):
                pass

        self.apply(_remove_weight_norm)

    def reset_parameters(self):
        """Reset parameters.
        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py
        """
        # 定义参数为float的正态分布。
        dist = paddle.distribution.Normal(loc=0.0, scale=0.02)

        def _reset_parameters(m):
            if isinstance(m, nn.Conv1D) or isinstance(m, nn.Conv1DTranspose):
                w = dist.sample(m.weight.shape)
                m.weight.set_value(w)

        self.apply(_reset_parameters)

    def inference(self, c):
        """Perform inference.
        Args:
            c (Tensor): 
                Input tensor (T, in_channels).
        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).
        """
        # (1, in_channels, T)
        c = c.transpose([1, 0]).unsqueeze(0)
        c_shape = paddle.shape(c)
        # prepare noise input
        # there is a bug in Paddle int division, we must convert a int tensor to int here
        noise_T = paddle.cast(paddle.ceil(c_shape[2] /
                                          int(self.noise_upsample_factor)),
                              dtype='int64')
        noise_size = (1, self.in_channels, noise_T)
        # (1, in_channels, T/noise_upsample_factor)
        noise = paddle.randn(noise_size)
        # (1, in_channels, T)
        x = self.noise_upsample(noise)
        x_shape = paddle.shape(x)
        total_length = c_shape[2] * self.upsample_factor
        # Dygraph to Static Graph bug here, 2021.12.15
        c = F.pad(c, (0, x_shape[2] - c_shape[2]),
                  "replicate",
                  data_format="NCL")
        # c.shape[2] == x.shape[2] here
        # (1, in_channels, T*prod(upsample_scales))
        for block in self.blocks:
            x, c = block(x, c)
        x = self.output_conv(x)[..., :total_length]
        return x.squeeze(0).transpose([1, 0])


class StyleMelGANDiscriminator(nn.Layer):
    """Style MelGAN disciminator module."""
    def __init__(
        self,
        repeats: int = 2,
        window_sizes: List[int] = [512, 1024, 2048, 4096],
        pqmf_params: List[List[int]] = [
            [1, None, None, None],
            [2, 62, 0.26700, 9.0],
            [4, 62, 0.14200, 9.0],
            [8, 62, 0.07949, 9.0],
        ],
        discriminator_params: Dict[str, Any] = {
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 16,
            "max_downsample_channels": 512,
            "bias": True,
            "downsample_scales": [4, 4, 4, 1],
            "nonlinear_activation": "leakyrelu",
            "nonlinear_activation_params": {
                "negative_slope": 0.2
            },
            "pad": "Pad1D",
            "pad_params": {
                "mode": "reflect"
            },
        },
        use_weight_norm: bool = True,
        init_type: str = "xavier_uniform",
    ):
        """Initilize Style MelGAN discriminator.

        Args:
            repeats (int): 
                Number of repititons to apply RWD.
            window_sizes (list): 
                List of random window sizes.
            pqmf_params (list): 
                List of list of Parameters for PQMF modules
            discriminator_params (dict): 
                Parameters for base discriminator module.
            use_weight_nom (bool): 
                Whether to apply weight normalization.
        """
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        # window size check
        assert len(window_sizes) == len(pqmf_params)
        sizes = [ws // p[0] for ws, p in zip(window_sizes, pqmf_params)]
        assert len(window_sizes) == sum([sizes[0] == size for size in sizes])

        self.repeats = repeats
        self.window_sizes = window_sizes
        self.pqmfs = nn.LayerList()
        self.discriminators = nn.LayerList()
        for pqmf_param in pqmf_params:
            d_params = copy.deepcopy(discriminator_params)
            d_params["in_channels"] = pqmf_param[0]
            if pqmf_param[0] == 1:
                self.pqmfs.append(nn.Identity())
            else:
                self.pqmfs.append(PQMF(*pqmf_param))
            self.discriminators.append(BaseDiscriminator(**d_params))

        nn.initializer.set_global_initializer(None)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): 
                Input tensor (B, 1, T).
        Returns:
            List: List of discriminator outputs, #items in the list will be
                equal to repeats * #discriminators.
        """
        outs = []
        for _ in range(self.repeats):
            outs += self._forward(x)
        return outs

    def _forward(self, x):
        outs = []
        for idx, (ws, pqmf, disc) in enumerate(
                zip(self.window_sizes, self.pqmfs, self.discriminators)):
            start_idx = int(np.random.randint(paddle.shape(x)[-1] - ws))
            x_ = x[:, :, start_idx:start_idx + ws]
            if idx == 0:
                # nn.Identity()
                x_ = pqmf(x_)
            else:
                x_ = pqmf.analysis(x_)
            outs += [disc(x_)]
        return outs

    def apply_weight_norm(self):
        """Recursively apply weight normalization to all the Convolution layers
        in the sublayers.
        """
        def _apply_weight_norm(layer):
            if isinstance(layer, (nn.Conv1D, nn.Conv1DTranspose)):
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

    def reset_parameters(self):
        """Reset parameters.
        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py
        """
        # 定义参数为float的正态分布。
        dist = paddle.distribution.Normal(loc=0.0, scale=0.02)

        def _reset_parameters(m):
            if isinstance(m, nn.Conv1D) or isinstance(m, nn.Conv1DTranspose):
                w = dist.sample(m.weight.shape)
                m.weight.set_value(w)

        self.apply(_reset_parameters)


class StyleMelGANInference(nn.Layer):
    def __init__(self, normalizer, style_melgan_generator):
        super().__init__()
        self.normalizer = normalizer
        self.style_melgan_generator = style_melgan_generator

    def forward(self, logmel):
        normalized_mel = self.normalizer(logmel)
        wav = self.style_melgan_generator.inference(normalized_mel)
        return wav
