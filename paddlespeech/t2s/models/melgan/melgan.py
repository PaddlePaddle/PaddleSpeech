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
"""MelGAN Modules."""
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import paddle
from paddle import nn

from paddlespeech.t2s.modules.activation import get_activation
from paddlespeech.t2s.modules.causal_conv import CausalConv1D
from paddlespeech.t2s.modules.causal_conv import CausalConv1DTranspose
from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.pqmf import PQMF
from paddlespeech.t2s.modules.residual_stack import ResidualStack


class MelGANGenerator(nn.Layer):
    """MelGAN generator module."""

    def __init__(
            self,
            in_channels: int=80,
            out_channels: int=1,
            kernel_size: int=7,
            channels: int=512,
            bias: bool=True,
            upsample_scales: List[int]=[8, 8, 2, 2],
            stack_kernel_size: int=3,
            stacks: int=3,
            nonlinear_activation: str="leakyrelu",
            nonlinear_activation_params: Dict[str, Any]={"negative_slope": 0.2},
            pad: str="Pad1D",
            pad_params: Dict[str, Any]={"mode": "reflect"},
            use_final_nonlinear_activation: bool=True,
            use_weight_norm: bool=True,
            use_causal_conv: bool=False,
            init_type: str="xavier_uniform", ):
        """Initialize MelGANGenerator module.

        Args:
            in_channels (int): 
                Number of input channels.
            out_channels (int): 
                Number of output channels,
                the number of sub-band is out_channels in multi-band melgan.
            kernel_size (int): 
                Kernel size of initial and final conv layer.
            channels (int): 
                Initial number of channels for conv layer.
            bias (bool): 
                Whether to add bias parameter in convolution layers.
            upsample_scales (List[int]): 
                List of upsampling scales.
            stack_kernel_size (int): 
                Kernel size of dilated conv layers in residual stack.
            stacks (int): 
                Number of stacks in a single residual stack.
            nonlinear_activation (Optional[str], optional): 
                Non linear activation in upsample network, by default None
            nonlinear_activation_params (Dict[str, Any], optional): 
                Parameters passed to the linear activation in the upsample network, by default {}
            pad (str): 
                Padding function module name before dilated convolution layer.
            pad_params (dict): 
                Hyperparameters for padding function.
            use_final_nonlinear_activation (nn.Layer): 
                Activation function for the final layer.
            use_weight_norm (bool): 
                Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool):
                Whether to use causal convolution.
        """
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        # for compatibility
        if nonlinear_activation:
            nonlinear_activation = nonlinear_activation.lower()

        # check hyper parameters is valid
        assert channels >= np.prod(upsample_scales)
        assert channels % (2**len(upsample_scales)) == 0
        if not use_causal_conv:
            assert (kernel_size - 1
                    ) % 2 == 0, "Not support even number kernel size."

        layers = []
        if not use_causal_conv:
            layers += [
                getattr(paddle.nn, pad)((kernel_size - 1) // 2, **pad_params),
                nn.Conv1D(in_channels, channels, kernel_size, bias_attr=bias),
            ]
        else:
            layers += [
                CausalConv1D(
                    in_channels,
                    channels,
                    kernel_size,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params, ),
            ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [
                get_activation(nonlinear_activation,
                               **nonlinear_activation_params)
            ]
            if not use_causal_conv:
                layers += [
                    nn.Conv1DTranspose(
                        channels // (2**i),
                        channels // (2**(i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        padding=upsample_scale // 2 + upsample_scale % 2,
                        output_padding=upsample_scale % 2,
                        bias_attr=bias, )
                ]
            else:
                layers += [
                    CausalConv1DTranspose(
                        channels // (2**i),
                        channels // (2**(i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        bias=bias, )
                ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2**(i + 1)),
                        dilation=stack_kernel_size**j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv, )
                ]

        # add final layer
        layers += [
            get_activation(nonlinear_activation, **nonlinear_activation_params)
        ]
        if not use_causal_conv:
            layers += [
                getattr(nn, pad)((kernel_size - 1) // 2, **pad_params),
                nn.Conv1D(
                    channels // (2**(i + 1)),
                    out_channels,
                    kernel_size,
                    bias_attr=bias),
            ]
        else:
            layers += [
                CausalConv1D(
                    channels // (2**(i + 1)),
                    out_channels,
                    kernel_size,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params, ),
            ]
        if use_final_nonlinear_activation:
            layers += [nn.Tanh()]

        # define the model as a single function
        self.melgan = nn.Sequential(*layers)
        nn.initializer.set_global_initializer(None)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

        # initialize pqmf for multi-band melgan inference
        if out_channels > 1:
            self.pqmf = PQMF(subbands=out_channels)
        else:
            self.pqmf = None

    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): 
                Input tensor (B, in_channels, T).
        Returns:
            Tensor: Output tensor (B, out_channels, T ** prod(upsample_scales)).
        """
        out = self.melgan(c)
        return out

    def apply_weight_norm(self):
        """Recursively apply weight normalization to all the Convolution layers
        in the sublayers.
        """

        def _apply_weight_norm(layer):
            if isinstance(layer, (nn.Conv1D, nn.Conv2D, nn.Conv1DTranspose)):
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

    def inference(self, c):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): 
                Input tensor (T, in_channels).
        Returns:
            Tensor: Output tensor (out_channels*T ** prod(upsample_scales), 1).
        """
        # pseudo batch
        c = c.transpose([1, 0]).unsqueeze(0)
        # (B, out_channels, T ** prod(upsample_scales)
        out = self.melgan(c)
        if self.pqmf is not None:
            # (B, 1, out_channels * T ** prod(upsample_scales)
            out = self.pqmf(out)
        out = out.squeeze(0).transpose([1, 0])
        return out


class MelGANDiscriminator(nn.Layer):
    """MelGAN discriminator module."""

    def __init__(
            self,
            in_channels: int=1,
            out_channels: int=1,
            kernel_sizes: List[int]=[5, 3],
            channels: int=16,
            max_downsample_channels: int=1024,
            bias: bool=True,
            downsample_scales: List[int]=[4, 4, 4, 4],
            nonlinear_activation: str="leakyrelu",
            nonlinear_activation_params: Dict[str, Any]={"negative_slope": 0.2},
            pad: str="Pad1D",
            pad_params: Dict[str, Any]={"mode": "reflect"},
            init_type: str="xavier_uniform", ):
        """Initilize MelGAN discriminator module.

        Args:
            in_channels (int): 
                Number of input channels.
            out_channels (int): 
                Number of output channels.
            kernel_sizes (List[int]): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15,
                the last two layers' kernel size will be 5 and 3, respectively.
            channels (int): 
                Initial number of channels for conv layer.
            max_downsample_channels (int): 
                Maximum number of channels for downsampling layers.
            bias (bool): 
                Whether to add bias parameter in convolution layers.
            downsample_scales (List[int]): 
                List of downsampling scales.
            nonlinear_activation (str): 
                Activation function module name.
            nonlinear_activation_params (dict): 
                Hyperparameters for activation function.
            pad (str): 
                Padding function module name before dilated convolution layer.
            pad_params (dict): 
                Hyperparameters for padding function.
        """
        super().__init__()

        # for compatibility
        if nonlinear_activation:
            nonlinear_activation = nonlinear_activation.lower()

        # initialize parameters
        initialize(self, init_type)

        self.layers = nn.LayerList()

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        self.layers.append(
            nn.Sequential(
                getattr(nn, pad)((np.prod(kernel_sizes) - 1) // 2, **
                                 pad_params),
                nn.Conv1D(
                    in_channels,
                    channels,
                    int(np.prod(kernel_sizes)),
                    bias_attr=bias),
                get_activation(nonlinear_activation, **
                               nonlinear_activation_params), ))

        # add downsample layers
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers.append(
                nn.Sequential(
                    nn.Conv1D(
                        in_chs,
                        out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 5,
                        groups=in_chs // 4,
                        bias_attr=bias, ),
                    get_activation(nonlinear_activation, **
                                   nonlinear_activation_params), ))
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers.append(
            nn.Sequential(
                nn.Conv1D(
                    in_chs,
                    out_chs,
                    kernel_sizes[0],
                    padding=(kernel_sizes[0] - 1) // 2,
                    bias_attr=bias, ),
                get_activation(nonlinear_activation, **
                               nonlinear_activation_params), ))
        self.layers.append(
            nn.Conv1D(
                out_chs,
                out_channels,
                kernel_sizes[1],
                padding=(kernel_sizes[1] - 1) // 2,
                bias_attr=bias, ), )

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): 
                Input noise signal (B, 1, T).
        Returns:
            List: List of output tensors of each layer (for feat_match_loss).
        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs


class MelGANMultiScaleDiscriminator(nn.Layer):
    """MelGAN multi-scale discriminator module."""

    def __init__(
            self,
            in_channels: int=1,
            out_channels: int=1,
            scales: int=3,
            downsample_pooling: str="AvgPool1D",
            # follow the official implementation setting
            downsample_pooling_params: Dict[str, Any]={
                "kernel_size": 4,
                "stride": 2,
                "padding": 1,
                "exclusive": True,
            },
            kernel_sizes: List[int]=[5, 3],
            channels: int=16,
            max_downsample_channels: int=1024,
            bias: bool=True,
            downsample_scales: List[int]=[4, 4, 4, 4],
            nonlinear_activation: str="leakyrelu",
            nonlinear_activation_params: Dict[str, Any]={"negative_slope": 0.2},
            pad: str="Pad1D",
            pad_params: Dict[str, Any]={"mode": "reflect"},
            use_weight_norm: bool=True,
            init_type: str="xavier_uniform", ):
        """Initilize MelGAN multi-scale discriminator module.

        Args:
            in_channels (int): 
                Number of input channels.
            out_channels (int): 
                Number of output channels.
            scales (int): 
                Number of multi-scales.
            downsample_pooling (str): 
                Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): 
                Parameters for the above pooling module.
            kernel_sizes (List[int]): 
                List of two kernel sizes. The sum will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
            channels (int): 
                Initial number of channels for conv layer.
            max_downsample_channels (int): 
                Maximum number of channels for downsampling layers.
            bias (bool): 
                Whether to add bias parameter in convolution layers.
            downsample_scales (List[int]): 
                List of downsampling scales.
            nonlinear_activation (str): 
                Activation function module name.
            nonlinear_activation_params (dict): 
                Hyperparameters for activation function.
            pad (str): 
                Padding function module name before dilated convolution layer.
            pad_params (dict): 
                Hyperparameters for padding function.
            use_causal_conv (bool): 
                Whether to use causal convolution.
        """
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        # for
        if nonlinear_activation:
            nonlinear_activation = nonlinear_activation.lower()

        self.discriminators = nn.LayerList()

        # add discriminators
        for _ in range(scales):
            self.discriminators.append(
                MelGANDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    bias=bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    pad=pad,
                    pad_params=pad_params, ))
        self.pooling = getattr(nn, downsample_pooling)(
            **downsample_pooling_params)

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
                Input noise signal (B, 1, T).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)

        return outs

    def apply_weight_norm(self):
        """Recursively apply weight normalization to all the Convolution layers
        in the sublayers.
        """

        def _apply_weight_norm(layer):
            if isinstance(layer, (nn.Conv1D, nn.Conv2D, nn.Conv1DTranspose)):
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


class MelGANInference(nn.Layer):
    def __init__(self, normalizer, melgan_generator):
        super().__init__()
        self.normalizer = normalizer
        self.melgan_generator = melgan_generator

    def forward(self, logmel):
        normalized_mel = self.normalizer(logmel)
        wav = self.melgan_generator.inference(normalized_mel)
        return wav
