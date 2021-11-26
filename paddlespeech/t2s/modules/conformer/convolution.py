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
"""ConvolutionModule definition."""
from paddle import nn


class ConvolutionModule(nn.Layer):
    """ConvolutionModule in Conformer model.
    Parameters
    ----------
    channels : int
        The number of channels of conv layers.
    kernel_size : int
        Kernerl size of conv layers.
    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super().__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1D(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=bias, )
        self.depthwise_conv = nn.Conv1D(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias_attr=bias, )
        self.norm = nn.BatchNorm1D(channels)
        self.pointwise_conv2 = nn.Conv1D(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=bias, )
        self.activation = activation

    def forward(self, x):
        """Compute convolution module.
        Parameters
        ----------
        x : paddle.Tensor
            Input tensor (#batch, time, channels).
        Returns
        ----------
        paddle.Tensor
            Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose([0, 2, 1])

        # GLU mechanism
        # (batch, 2*channel, time)
        x = self.pointwise_conv1(x)
        # (batch, channel, time)
        x = nn.functional.glu(x, axis=1)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose([0, 2, 1])
