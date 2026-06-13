# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

"""ConvolutionModule definition."""
from typing import Tuple


class ConvolutionModule(paddle.nn.Layer):
    """ConvolutionModule in Conformer model."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: paddle.nn.Layer = paddle.nn.ReLU(),
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
    ):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()
        self.pointwise_conv1 = paddle.nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = paddle.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        assert norm in ["batch_norm", "layer_norm"]
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = paddle.nn.BatchNorm1D(num_features=channels)
        else:
            self.use_layer_norm = True
            self.norm = paddle.nn.LayerNorm(normalized_shape=channels)
        self.pointwise_conv2 = paddle.nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.activation = activation

    def forward(
        self,
        x: paddle.Tensor,
        mask_pad: paddle.Tensor = paddle.ones((0, 0, 0), dtype=paddle.bool),
        cache: paddle.Tensor = paddle.zeros((0, 0, 0)),
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)
        if self.lorder > 0:
            if cache.size(2) == 0:
                x = paddle.compat.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                assert cache.size(0) == x.size(0)
                assert cache.size(1) == x.size(1)
                x = paddle.cat((cache, x), dim=2)
            assert x.size(2) > self.lorder
            new_cache = x[:, :, -self.lorder :]
        else:
            new_cache = paddle.zeros((0, 0, 0), dtype=x.dtype, device=x.place)
        x = self.pointwise_conv1(x)
        x = paddle.nn.functional.glu(x=x, axis=1)
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)
        return x.transpose(1, 2), new_cache
