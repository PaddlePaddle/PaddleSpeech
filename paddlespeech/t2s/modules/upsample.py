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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from paddle import nn
from paddle.nn import functional as F

from paddlespeech.t2s.modules.activation import get_activation


class Stretch2D(nn.Layer):
    def __init__(self, w_scale: int, h_scale: int, mode: str = "nearest"):
        """Strech an image (or image-like object) with some interpolation.

        Args:
            w_scale (int): 
                Scalar of width.
            h_scale (int): 
                Scalar of the height.
            mode (str, optional): 
                Interpolation mode, modes suppored are "nearest", "bilinear", 
                "trilinear", "bicubic", "linear" and "area",by default "nearest"
        For more details about interpolation, see 
            `paddle.nn.functional.interpolate <https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/nn/functional/interpolate_en.html>`_.
        """
        super().__init__()
        self.w_scale = w_scale
        self.h_scale = h_scale
        self.mode = mode

    def forward(self, x):
        """

        Args: 
            x (Tensor): 
                Shape (N, C, H, W)

        Returns:
            Tensor: 
                The stretched image. Shape (N, C, H', W'), where ``H'=h_scale * H``, ``W'=w_scale * W``.
            
        """
        out = F.interpolate(x,
                            scale_factor=(self.h_scale, self.w_scale),
                            mode=self.mode)
        return out


class UpsampleNet(nn.Layer):
    """A Layer to upsample spectrogram by applying consecutive stretch and
    convolutions.

    Args:
        upsample_scales (List[int]): 
            Upsampling factors for each strech.
        nonlinear_activation (Optional[str], optional): 
            Activation after each convolution, by default None
        nonlinear_activation_params (Dict[str, Any], optional): 
            Parameters passed to construct the activation, by default {}
        interpolate_mode (str, optional): 
            Interpolation mode of the strech, by default "nearest"
        freq_axis_kernel_size (int, optional): 
            Convolution kernel size along the frequency axis, by default 1
        use_causal_conv (bool, optional): 
            Whether to use causal padding before convolution, by default False
            If True, Causal padding is used along the time axis, 
            i.e. padding amount is ``receptive field - 1`` and 0 for before and after, respectively.
            If False, "same" padding is used along the time axis.
    """
    def __init__(self,
                 upsample_scales: List[int],
                 nonlinear_activation: Optional[str] = None,
                 nonlinear_activation_params: Dict[str, Any] = {},
                 interpolate_mode: str = "nearest",
                 freq_axis_kernel_size: int = 1,
                 use_causal_conv: bool = False):
        super().__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = nn.LayerList()

        for scale in upsample_scales:
            stretch = Stretch2D(scale, 1, interpolate_mode)
            assert freq_axis_kernel_size % 2 == 1
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = (freq_axis_kernel_size, scale * 2 + 1)
            if use_causal_conv:
                padding = (freq_axis_padding, scale * 2)
            else:
                padding = (freq_axis_padding, scale)
            conv = nn.Conv2D(1,
                             1,
                             kernel_size,
                             padding=padding,
                             bias_attr=False)
            self.up_layers.extend([stretch, conv])
            if nonlinear_activation is not None:
                # for compatibility
                nonlinear_activation = nonlinear_activation.lower()

                nonlinear = get_activation(nonlinear_activation,
                                           **nonlinear_activation_params)
                self.up_layers.append(nonlinear)

    def forward(self, c):
        """
        Args:
            c (Tensor): 
                spectrogram. Shape (N, F, T)

        Returns: 
            Tensor: upsampled spectrogram.
                Shape (N, F, T'), where ``T' = upsample_factor * T``, 
        """
        c = c.unsqueeze(1)
        for f in self.up_layers:
            if self.use_causal_conv and isinstance(f, nn.Conv2D):
                c = f(c)[:, :, :, c.shape[-1]]
            else:
                c = f(c)
        return c.squeeze(1)


class ConvInUpsampleNet(nn.Layer):
    """A Layer to upsample spectrogram composed of a convolution and an 
    UpsampleNet.
    
    Args:
        upsample_scales (List[int]): 
            Upsampling factors for each strech.
        nonlinear_activation (Optional[str], optional): 
            Activation after each convolution, by default None
        nonlinear_activation_params (Dict[str, Any], optional): 
            Parameters passed to construct the activation, by default {}
        interpolate_mode (str, optional): 
            Interpolation mode of the strech, by default "nearest"
        freq_axis_kernel_size (int, optional): 
            Convolution kernel size along the frequency axis, by default 1
        aux_channels (int, optional): 
            Feature size of the input, by default 80
        aux_context_window (int, optional): 
            Context window of the first 1D convolution applied to the input. It 
            related to the kernel size of the convolution, by default 0
            If use causal convolution, the kernel size is ``window + 1``, 
            else the kernel size is ``2 * window + 1``.
        use_causal_conv (bool, optional):
            Whether to use causal padding before convolution, by default False
            If True, Causal padding is used along the time axis, i.e. padding 
            amount is ``receptive field - 1`` and 0 for before and after, respectively.
            If False, "same" padding is used along the time axis.
    """
    def __init__(self,
                 upsample_scales: List[int],
                 nonlinear_activation: Optional[str] = None,
                 nonlinear_activation_params: Dict[str, Any] = {},
                 interpolate_mode: str = "nearest",
                 freq_axis_kernel_size: int = 1,
                 aux_channels: int = 80,
                 aux_context_window: int = 0,
                 use_causal_conv: bool = False):
        super().__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        kernel_size = aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        self.conv_in = nn.Conv1D(aux_channels,
                                 aux_channels,
                                 kernel_size=kernel_size,
                                 bias_attr=False)
        self.upsample = UpsampleNet(
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            use_causal_conv=use_causal_conv)

    def forward(self, c):
        """
        Args:
            c (Tensor): 
                spectrogram. Shape (N, F, T)

        Returns:
            Tensors: upsampled spectrogram. Shape (N, F, T'), where ``T' = upsample_factor * T``, 
        """
        c_ = self.conv_in(c)
        c = c_[:, :, :-self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)
