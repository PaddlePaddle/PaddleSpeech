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
# This code is based on https://github.com/jik876/hifi-gan.
import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.t2s.modules.activation import get_activation
from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.residual_block import HiFiGANResidualBlock as ResidualBlock


class HiFiGANGenerator(nn.Layer):
    """HiFiGAN generator module."""

    def __init__(
            self,
            in_channels: int=80,
            out_channels: int=1,
            channels: int=512,
            global_channels: int=-1,
            kernel_size: int=7,
            upsample_scales: List[int]=(8, 8, 2, 2),
            upsample_kernel_sizes: List[int]=(16, 16, 4, 4),
            resblock_kernel_sizes: List[int]=(3, 7, 11),
            resblock_dilations: List[List[int]]=[(1, 3, 5), (1, 3, 5),
                                                 (1, 3, 5)],
            use_additional_convs: bool=True,
            bias: bool=True,
            nonlinear_activation: str="leakyrelu",
            nonlinear_activation_params: Dict[str, Any]={"negative_slope": 0.1},
            use_weight_norm: bool=True,
            init_type: str="xavier_uniform", ):
        """Initialize HiFiGANGenerator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            global_channels (int): Number of global conditioning channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = nn.Conv1D(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2, )
        self.upsamples = nn.LayerList()
        self.blocks = nn.LayerList()
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples.append(
                nn.Sequential(
                    get_activation(nonlinear_activation, **
                                   nonlinear_activation_params),
                    nn.Conv1DTranspose(
                        channels // (2**i),
                        channels // (2**(i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] // 2 + upsample_scales[i] %
                        2,
                        output_padding=upsample_scales[i] % 2, ), ))
            for j in range(len(resblock_kernel_sizes)):
                self.blocks.append(
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2**(i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    ))
        self.output_conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1D(
                channels // (2**(i + 1)),
                out_channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2, ),
            nn.Tanh(), )

        if global_channels > 0:
            self.global_conv = nn.Conv1D(global_channels, channels, 1)

        nn.initializer.set_global_initializer(None)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c, g: Optional[paddle.Tensor]=None):
        """Calculate forward propagation.
        
        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        c = self.input_conv(c)
        if g is not None:
            c = c + self.global_conv(g)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            # initialize
            cs = 0.0
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c

    def reset_parameters(self):
        """Reset parameters.
        This initialization follows official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py
        """
        # 定义参数为float的正态分布。
        dist = paddle.distribution.Normal(loc=0.0, scale=0.01)

        def _reset_parameters(m):
            if isinstance(m, nn.Conv1D) or isinstance(m, nn.Conv1DTranspose):
                w = dist.sample(m.weight.shape)
                m.weight.set_value(w)

        self.apply(_reset_parameters)

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

    def inference(self, c, g: Optional[paddle.Tensor]=None):
        """Perform inference.
        Args:
            c (Tensor): Input tensor (T, in_channels).
                normalize_before (bool): Whether to perform normalization.
            g (Optional[Tensor]): Global conditioning tensor (global_channels, 1).
        Returns:
            Tensor:
                Output tensor (T ** prod(upsample_scales), out_channels).
        """
        if g is not None:
            g = g.unsqueeze(0)
        c = self.forward(c.transpose([1, 0]).unsqueeze(0), g=g)
        return c.squeeze(0).transpose([1, 0])


class HiFiGANPeriodDiscriminator(nn.Layer):
    """HiFiGAN period discriminator module."""

    def __init__(
            self,
            in_channels: int=1,
            out_channels: int=1,
            period: int=3,
            kernel_sizes: List[int]=[5, 3],
            channels: int=32,
            downsample_scales: List[int]=[3, 3, 3, 3, 1],
            max_downsample_channels: int=1024,
            bias: bool=True,
            nonlinear_activation: str="leakyrelu",
            nonlinear_activation_params: Dict[str, Any]={"negative_slope": 0.1},
            use_weight_norm: bool=True,
            use_spectral_norm: bool=False,
            init_type: str="xavier_uniform", ):
        """Initialize HiFiGANPeriodDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv layer.
            channels (int): Number of initial channels.
            downsample_scales (list): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = nn.LayerList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0), ),
                    get_activation(nonlinear_activation, **
                                   nonlinear_activation_params), ))
            in_chs = out_chs
            # NOTE: Use downsample_scale + 1?
            out_chs = min(out_chs * 4, max_downsample_channels)
        self.output_conv = nn.Conv2D(
            out_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0), )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).
        Returns:
            list: List of each layer's tensors.
        """
        # transform 1d to 2d -> (B, C, T/P, P)
        b, c, t = paddle.shape(x)
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect", data_format="NCL")
            t += n_pad
        x = x.reshape([b, c, t // self.period, self.period])

        # forward conv
        outs = []
        for layer in self.convs:
            x = layer(x)
            outs += [x]
        x = self.output_conv(x)
        x = paddle.flatten(x, 1, -1)
        outs += [x]

        return outs

    def apply_weight_norm(self):
        """Recursively apply weight normalization to all the Convolution layers
        in the sublayers.
        """

        def _apply_weight_norm(layer):
            if isinstance(layer, (nn.Conv1D, nn.Conv2D, nn.Conv1DTranspose)):
                nn.utils.weight_norm(layer)

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, nn.Conv2D):
                nn.utils.spectral_norm(m)

        self.apply(_apply_spectral_norm)


class HiFiGANMultiPeriodDiscriminator(nn.Layer):
    """HiFiGAN multi-period discriminator module."""

    def __init__(
            self,
            periods: List[int]=[2, 3, 5, 7, 11],
            discriminator_params: Dict[str, Any]={
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 32,
                "downsample_scales": [3, 3, 3, 3, 1],
                "max_downsample_channels": 1024,
                "bias": True,
                "nonlinear_activation": "leakyrelu",
                "nonlinear_activation_params": {
                    "negative_slope": 0.1
                },
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
            init_type: str="xavier_uniform", ):
        """Initialize HiFiGANMultiPeriodDiscriminator module.

        Args:
            periods (list): List of periods.
            discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.
        """
        super().__init__()
        # initialize parameters
        initialize(self, init_type)

        self.discriminators = nn.LayerList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params["period"] = period
            self.discriminators.append(HiFiGANPeriodDiscriminator(**params))

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]

        return outs


class HiFiGANScaleDiscriminator(nn.Layer):
    """HiFi-GAN scale discriminator module."""

    def __init__(
            self,
            in_channels: int=1,
            out_channels: int=1,
            kernel_sizes: List[int]=[15, 41, 5, 3],
            channels: int=128,
            max_downsample_channels: int=1024,
            max_groups: int=16,
            bias: bool=True,
            downsample_scales: List[int]=[2, 2, 4, 4, 1],
            nonlinear_activation: str="leakyrelu",
            nonlinear_activation_params: Dict[str, Any]={"negative_slope": 0.1},
            use_weight_norm: bool=True,
            use_spectral_norm: bool=False,
            init_type: str="xavier_uniform", ):
        """Initilize HiFiGAN scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of four kernel sizes. The first will be used for the first conv layer,
                and the second is for downsampling part, and the remaining two are for output layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        self.layers = nn.LayerList()

        # check kernel size is valid
        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1

        # add first layer
        self.layers.append(
            nn.Sequential(
                nn.Conv1D(
                    in_channels,
                    channels,
                    # NOTE: Use always the same kernel size
                    kernel_sizes[0],
                    bias_attr=bias,
                    padding=(kernel_sizes[0] - 1) // 2, ),
                get_activation(nonlinear_activation, **
                               nonlinear_activation_params), ))

        # add downsample layers
        in_chs = channels
        out_chs = channels
        # NOTE(kan-bayashi): Remove hard coding?
        groups = 4
        for downsample_scale in downsample_scales:
            self.layers.append(
                nn.Sequential(
                    nn.Conv1D(
                        in_chs,
                        out_chs,
                        kernel_size=kernel_sizes[1],
                        stride=downsample_scale,
                        padding=(kernel_sizes[1] - 1) // 2,
                        groups=groups,
                        bias_attr=bias, ),
                    get_activation(nonlinear_activation, **
                                   nonlinear_activation_params), ))
            in_chs = out_chs
            # NOTE: Remove hard coding?
            out_chs = min(in_chs * 2, max_downsample_channels)
            # NOTE: Remove hard coding?
            groups = min(groups * 4, max_groups)

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers.append(
            nn.Sequential(
                nn.Conv1D(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_sizes[2],
                    stride=1,
                    padding=(kernel_sizes[2] - 1) // 2,
                    bias_attr=bias, ),
                get_activation(nonlinear_activation, **
                               nonlinear_activation_params), ))
        self.layers.append(
            nn.Conv1D(
                out_chs,
                out_channels,
                kernel_size=kernel_sizes[3],
                stride=1,
                padding=(kernel_sizes[3] - 1) // 2,
                bias_attr=bias, ), )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of output tensors of each layer.
        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs

    def apply_weight_norm(self):
        """Recursively apply weight normalization to all the Convolution layers
        in the sublayers.
        """

        def _apply_weight_norm(layer):
            if isinstance(layer, (nn.Conv1D, nn.Conv2D, nn.Conv1DTranspose)):
                nn.utils.weight_norm(layer)

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, nn.Conv2D):
                nn.utils.spectral_norm(m)

        self.apply(_apply_spectral_norm)


class HiFiGANMultiScaleDiscriminator(nn.Layer):
    """HiFi-GAN multi-scale discriminator module."""

    def __init__(
            self,
            scales: int=3,
            downsample_pooling: str="AvgPool1D",
            # follow the official implementation setting
            downsample_pooling_params: Dict[str, Any]={
                "kernel_size": 4,
                "stride": 2,
                "padding": 2,
            },
            discriminator_params: Dict[str, Any]={
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [15, 41, 5, 3],
                "channels": 128,
                "max_downsample_channels": 1024,
                "max_groups": 16,
                "bias": True,
                "downsample_scales": [2, 2, 4, 4, 1],
                "nonlinear_activation": "leakyrelu",
                "nonlinear_activation_params": {
                    "negative_slope": 0.1
                },
            },
            follow_official_norm: bool=False,
            init_type: str="xavier_uniform", ):
        """Initilize HiFiGAN multi-scale discriminator module.
   
        Args:
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other discriminators use weight norm.
        """
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        self.discriminators = nn.LayerList()

        # add discriminators
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                if i == 0:
                    params["use_weight_norm"] = False
                    params["use_spectral_norm"] = True
                else:
                    params["use_weight_norm"] = True
                    params["use_spectral_norm"] = False
            self.discriminators.append(HiFiGANScaleDiscriminator(**params))
        self.pooling = getattr(nn, downsample_pooling)(
            **downsample_pooling_params)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)

        return outs


class HiFiGANMultiScaleMultiPeriodDiscriminator(nn.Layer):
    """HiFi-GAN multi-scale + multi-period discriminator module."""

    def __init__(
            self,
            # Multi-scale discriminator related
            scales: int=3,
            scale_downsample_pooling: str="AvgPool1D",
            scale_downsample_pooling_params: Dict[str, Any]={
                "kernel_size": 4,
                "stride": 2,
                "padding": 2,
            },
            scale_discriminator_params: Dict[str, Any]={
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [15, 41, 5, 3],
                "channels": 128,
                "max_downsample_channels": 1024,
                "max_groups": 16,
                "bias": True,
                "downsample_scales": [2, 2, 4, 4, 1],
                "nonlinear_activation": "leakyrelu",
                "nonlinear_activation_params": {
                    "negative_slope": 0.1
                },
            },
            follow_official_norm: bool=True,
            # Multi-period discriminator related
            periods: List[int]=[2, 3, 5, 7, 11],
            period_discriminator_params: Dict[str, Any]={
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 32,
                "downsample_scales": [3, 3, 3, 3, 1],
                "max_downsample_channels": 1024,
                "bias": True,
                "nonlinear_activation": "leakyrelu",
                "nonlinear_activation_params": {
                    "negative_slope": 0.1
                },
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
            init_type: str="xavier_uniform", ):
        """Initilize HiFiGAN multi-scale + multi-period discriminator module.

        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm （bool): Whether to follow the norm setting of the official implementaion. 
                The first discriminator uses spectral norm and the other discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.
        """
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        self.msd = HiFiGANMultiScaleDiscriminator(
            scales=scales,
            downsample_pooling=scale_downsample_pooling,
            downsample_pooling_params=scale_downsample_pooling_params,
            discriminator_params=scale_discriminator_params,
            follow_official_norm=follow_official_norm, )
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params, )

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List:
                List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.
        """
        msd_outs = self.msd(x)
        mpd_outs = self.mpd(x)
        return msd_outs + mpd_outs


class HiFiGANInference(nn.Layer):
    def __init__(self, normalizer, hifigan_generator):
        super().__init__()
        self.normalizer = normalizer
        self.hifigan_generator = hifigan_generator

    def forward(self, logmel):
        normalized_mel = self.normalizer(logmel)
        wav = self.hifigan_generator.inference(normalized_mel)
        return wav
