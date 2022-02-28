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

"""
customized layers
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlespeech.vector.utils.data_utils import length_to_mask


class StatisticsPooling(nn.Layer):
    """
    Stats pooling layer
    """

    def __init__(self):
        super(StatisticsPooling, self).__init__()
        self.eps = 1e-6

    def forward(self, x, lengths=None):  # size: (batch_size, dim, T)
        """
        Forward inference for statistics layer

        Args:
            x: input tensor, shape=(batch_size, dim, T)

        Returns:
            res: output tensor
        """
        if lengths is None:
            mean = paddle.mean(x, axis=2)
            std = paddle.std(x, axis=2)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                # Avoiding padded time steps
                actual_size = int(paddle.round(paddle.to_tensor(lengths[snt_id] * x.shape[2], dtype="float32")))

                # computing statistics
                mean.append(
                    paddle.mean(x[snt_id, :, 0: actual_size - 1], axis=1)
                )
                std.append(
                    paddle.std(x[snt_id, :, 0: actual_size - 1], axis=1)
                )

            mean = paddle.stack(mean)
            std = paddle.stack(std)

        gnoise = self._get_gauss_noise(mean.shape)
        mean += gnoise
        std = std + self.eps
        res = paddle.concat((mean, std), axis=1)

        return res

    def _get_gauss_noise(self, shape_of_tensor):
        """Returns a tensor of epsilon Gaussian noise.

        Args:
            shape_of_tensor : tensor, It represents the size of tensor for generating Gaussian noise.
        """
        gnoise = paddle.randn(shape_of_tensor)
        gnoise -= paddle.min(gnoise)
        gnoise /= paddle.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)

        return gnoise


class SEBlock(nn.Layer):
    """An implementation of squeeuze-and-excitation block.

    Args:
        in_channels : int. The number of input channels.
        se_channels : int. The number of output channels after squeeze.
        out_channels : int. The number of output channels.
    """

    def __init__(self, in_channels, se_channels, out_channels=None):
        super(SEBlock, self).__init__()

        if not out_channels:
            out_channels = in_channels
        self.conv1 = nn.Conv1D(in_channels, se_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1D(se_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        """
        Forward propagationg
        """
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L)
            mask = mask.unsqueeze(1)
            total = mask.sum(axis=2, keepdim=True)
            s = (x * mask).sum(axis=2, keepdim=True) / total
        else:
            s = x.mean(axis=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x


class AdditiveMarginLinear(nn.Layer):
    """
    Additive Margin Llinear layer
    """
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 name=None):
        super(AdditiveMarginLinear, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self.in_features = in_features
        self.out_features = out_features
        self._weight_attr = paddle.framework.ParamAttr(
                name="linear_weight",
                initializer=paddle.nn.initializer.XavierNormal())
        self.weight = self.create_parameter(
                shape=[in_features, out_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False)
        self.name = name

    def forward(self, x):
        """
        Forward propagationg

        Args:
            x: input features, size = (B, F), F is feature len

        Returns:
            cos_theta: output tensor. Angle between input embedding and weiths
        """
        assert x.shape[1] == self.in_features
        x_norm = F.normalize(x)
        w_norm = F.normalize(self.weight, p=2, axis=0)
        cos_theta = F.linear(x_norm, w_norm)

        return cos_theta
