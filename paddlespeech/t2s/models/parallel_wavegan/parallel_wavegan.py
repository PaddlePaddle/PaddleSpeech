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
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import paddle
from paddle import nn

from paddlespeech.t2s.modules.activation import get_activation
from paddlespeech.t2s.modules.nets_utils import initialize
from paddlespeech.t2s.modules.residual_block import WaveNetResidualBlock as ResidualBlock
from paddlespeech.t2s.modules.upsample import ConvInUpsampleNet


class PWGGenerator(nn.Layer):
    """Wave Generator for Parallel WaveGAN

    Args:
        in_channels (int, optional): 
            Number of channels of the input waveform, by default 1
        out_channels (int, optional): 
            Number of channels of the output waveform, by default 1
        kernel_size (int, optional): 
            Kernel size of the residual blocks inside, by default 3
        layers (int, optional): 
            Number of residual blocks inside, by default 30
        stacks (int, optional):
            The number of groups to split the residual blocks into, by default 3
            Within each group, the dilation of the residual block grows exponentially.
        residual_channels (int, optional): 
            Residual channel of the residual blocks, by default 64
        gate_channels (int, optional): 
            Gate channel of the residual blocks, by default 128
        skip_channels (int, optional): 
            Skip channel of the residual blocks, by default 64
        aux_channels (int, optional): 
            Auxiliary channel of the residual blocks, by default 80
        aux_context_window (int, optional): 
            The context window size of the first convolution applied to the auxiliary input, by default 2
        dropout (float, optional): 
            Dropout of the residual blocks, by default 0.
        bias (bool, optional): 
            Whether to use bias in residual blocks, by default True
        use_weight_norm (bool, optional): 
            Whether to use weight norm in all convolutions, by default True
        use_causal_conv (bool, optional): 
            Whether to use causal padding in the upsample network and residual blocks, by default False
        upsample_scales (List[int], optional): 
            Upsample scales of the upsample network, by default [4, 4, 4, 4]
        nonlinear_activation (Optional[str], optional): 
            Non linear activation in upsample network, by default None
        nonlinear_activation_params (Dict[str, Any], optional): 
            Parameters passed to the linear activation in the upsample network, by default {}
        interpolate_mode (str, optional): 
            Interpolation mode of the upsample network, by default "nearest"
        freq_axis_kernel_size (int, optional): 
            Kernel size along the frequency axis of the upsample network, by default 1
    """

    def __init__(
            self,
            in_channels: int=1,
            out_channels: int=1,
            kernel_size: int=3,
            layers: int=30,
            stacks: int=3,
            residual_channels: int=64,
            gate_channels: int=128,
            skip_channels: int=64,
            aux_channels: int=80,
            aux_context_window: int=2,
            dropout: float=0.,
            bias: bool=True,
            use_weight_norm: bool=True,
            use_causal_conv: bool=False,
            upsample_scales: List[int]=[4, 4, 4, 4],
            nonlinear_activation: Optional[str]=None,
            nonlinear_activation_params: Dict[str, Any]={},
            interpolate_mode: str="nearest",
            freq_axis_kernel_size: int=1,
            init_type: str="xavier_uniform", ):
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        # for compatibility
        if nonlinear_activation:
            nonlinear_activation = nonlinear_activation.lower()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size

        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        self.first_conv = nn.Conv1D(
            in_channels, residual_channels, 1, bias_attr=True)
        self.upsample_net = ConvInUpsampleNet(
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            aux_channels=aux_channels,
            aux_context_window=aux_context_window,
            use_causal_conv=use_causal_conv)
        self.upsample_factor = np.prod(upsample_scales)

        self.conv_layers = nn.LayerList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                use_causal_conv=use_causal_conv)
            self.conv_layers.append(conv)

        self.last_conv_layers = nn.Sequential(nn.ReLU(),
                                              nn.Conv1D(
                                                  skip_channels,
                                                  skip_channels,
                                                  1,
                                                  bias_attr=True),
                                              nn.ReLU(),
                                              nn.Conv1D(
                                                  skip_channels,
                                                  out_channels,
                                                  1,
                                                  bias_attr=True))

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c):
        """Generate waveform.

        Args:
            x(Tensor): 
                Shape (N, C_in, T), The input waveform.
            c(Tensor): 
                Shape (N, C_aux, T'). The auxiliary input (e.g. spectrogram). 
                It is upsampled to match the time resolution of the input.

        Returns:
            Tensor: Shape (N, C_out, T), the generated waveform.
        """
        c = self.upsample_net(c)
        assert c.shape[-1] == x.shape[-1]

        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
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

    def inference(self, c=None):
        """Waveform generation. This function is used for single instance inference.

        Args:
            c(Tensor, optional, optional): 
                Shape (T', C_aux), the auxiliary input, by default None
            x(Tensor, optional): 
                Shape (T, C_in), the noise waveform, by default None

        Returns:
            Tensor: Shape (T, C_out), the generated waveform
        """
        # when to static, can not input x, see https://github.com/PaddlePaddle/Parakeet/pull/132/files
        x = paddle.randn(
            [1, self.in_channels, paddle.shape(c)[0:1] * self.upsample_factor])
        c = paddle.transpose(c, [1, 0]).unsqueeze(0)  # pseudo batch
        c = nn.Pad1D(self.aux_context_window, mode='replicate')(c)
        out = self(x, c).squeeze(0).transpose([1, 0])
        return out


class PWGDiscriminator(nn.Layer):
    """A convolutional discriminator for audio.

    Args:
        in_channels (int, optional): 
            Number of channels of the input audio, by default 1
        out_channels (int, optional): 
            Output feature size, by default 1
        kernel_size (int, optional): 
            Kernel size of convolutional sublayers, by default 3
        layers (int, optional): 
            Number of layers, by default 10
        conv_channels (int, optional): 
            Feature size of the convolutional sublayers, by default 64
        dilation_factor (int, optional): 
            The factor with which dilation of each convolutional sublayers grows 
            exponentially if it is greater than 1, else the dilation of each convolutional sublayers grows linearly, 
            by default 1
        nonlinear_activation (str, optional): 
            The activation after each convolutional sublayer, by default "leakyrelu"
        nonlinear_activation_params (Dict[str, Any], optional): 
            The parameters passed to the activation's initializer, by default {"negative_slope": 0.2}
        bias (bool, optional): 
            Whether to use bias in convolutional sublayers, by default True
        use_weight_norm (bool, optional): 
            Whether to use weight normalization at all convolutional sublayers, by default True
    """

    def __init__(
            self,
            in_channels: int=1,
            out_channels: int=1,
            kernel_size: int=3,
            layers: int=10,
            conv_channels: int=64,
            dilation_factor: int=1,
            nonlinear_activation: str="leakyrelu",
            nonlinear_activation_params: Dict[str, Any]={"negative_slope": 0.2},
            bias: bool=True,
            use_weight_norm: bool=True,
            init_type: str="xavier_uniform", ):
        super().__init__()

        # initialize parameters
        initialize(self, init_type)
        # for compatibility
        if nonlinear_activation:
            nonlinear_activation = nonlinear_activation.lower()

        assert kernel_size % 2 == 1
        assert dilation_factor > 0
        conv_layers = []
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor**i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = nn.Conv1D(
                conv_in_channels,
                conv_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias_attr=bias)
            nonlinear = get_activation(nonlinear_activation,
                                       **nonlinear_activation_params)
            conv_layers.append(conv_layer)
            conv_layers.append(nonlinear)
        padding = (kernel_size - 1) // 2
        last_conv = nn.Conv1D(
            conv_in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias_attr=bias)
        conv_layers.append(last_conv)
        self.conv_layers = nn.Sequential(*conv_layers)

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """

        Args:
            x (Tensor): 
                Shape (N, in_channels, num_samples), the input audio.

        Returns:
            Tensor: Shape (N, out_channels, num_samples), the predicted logits.
        """
        return self.conv_layers(x)

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


class ResidualPWGDiscriminator(nn.Layer):
    """A wavenet-style discriminator for audio.

    Args:
        in_channels (int, optional): 
            Number of channels of the input audio, by default 1
        out_channels (int, optional): 
            Output feature size, by default 1
        kernel_size (int, optional): 
            Kernel size of residual blocks, by default 3
        layers (int, optional): 
            Number of residual blocks, by default 30
        stacks (int, optional): 
            Number of groups of residual blocks, within which the dilation 
            of each residual blocks grows exponentially, by default 3
        residual_channels (int, optional): 
            Residual channels of residual blocks, by default 64
        gate_channels (int, optional): 
            Gate channels of residual blocks, by default 128
        skip_channels (int, optional): 
            Skip channels of residual blocks, by default 64
        dropout (float, optional): 
            Dropout probability of residual blocks, by default 0.
        bias (bool, optional): 
            Whether to use bias in residual blocks, by default True
        use_weight_norm (bool, optional): 
            Whether to use weight normalization in all convolutional layers, by default True
        use_causal_conv (bool, optional): 
            Whether to use causal convolution in residual blocks, by default False
        nonlinear_activation (str, optional): 
            Activation after convolutions other than those in residual blocks, by default "leakyrelu"
        nonlinear_activation_params (Dict[str, Any], optional): 
            Parameters to pass to the activation, by default {"negative_slope": 0.2}
    """

    def __init__(
            self,
            in_channels: int=1,
            out_channels: int=1,
            kernel_size: int=3,
            layers: int=30,
            stacks: int=3,
            residual_channels: int=64,
            gate_channels: int=128,
            skip_channels: int=64,
            dropout: float=0.,
            bias: bool=True,
            use_weight_norm: bool=True,
            use_causal_conv: bool=False,
            nonlinear_activation: str="leakyrelu",
            nonlinear_activation_params: Dict[str, Any]={"negative_slope": 0.2},
            init_type: str="xavier_uniform", ):
        super().__init__()

        # initialize parameters
        initialize(self, init_type)

        # for compatibility
        if nonlinear_activation:
            nonlinear_activation = nonlinear_activation.lower()

        assert kernel_size % 2 == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size

        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        self.first_conv = nn.Sequential(
            nn.Conv1D(in_channels, residual_channels, 1, bias_attr=True),
            get_activation(nonlinear_activation, **nonlinear_activation_params))

        self.conv_layers = nn.LayerList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=None,  # no auxiliary input
                dropout=dropout,
                dilation=dilation,
                bias=bias,
                use_causal_conv=use_causal_conv)
            self.conv_layers.append(conv)

        self.last_conv_layers = nn.Sequential(
            get_activation(nonlinear_activation, **nonlinear_activation_params),
            nn.Conv1D(skip_channels, skip_channels, 1, bias_attr=True),
            get_activation(nonlinear_activation, **nonlinear_activation_params),
            nn.Conv1D(skip_channels, out_channels, 1, bias_attr=True))

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """
        Args:
            x(Tensor): 
                Shape (N, in_channels, num_samples), the input audio.↩

        Returns:
            Tensor: Shape (N, out_channels, num_samples), the predicted logits.
        """
        x = self.first_conv(x)
        skip = 0
        for f in self.conv_layers:
            x, h = f(x, None)
            skip += h
        skip *= math.sqrt(1 / len(self.conv_layers))

        x = skip
        x = self.last_conv_layers(x)
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


class PWGInference(nn.Layer):
    def __init__(self, normalizer, pwg_generator):
        super().__init__()
        self.normalizer = normalizer
        self.pwg_generator = pwg_generator

    def forward(self, logmel):
        normalized_mel = self.normalizer(logmel)
        wav = self.pwg_generator.inference(normalized_mel)
        return wav
