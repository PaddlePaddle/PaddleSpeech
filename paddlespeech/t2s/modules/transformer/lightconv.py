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
"""Lightweight Convolution Module."""
import numpy
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.t2s.modules.activation import get_activation
from paddlespeech.t2s.modules.masked_fill import masked_fill

MIN_VALUE = float(numpy.finfo(numpy.float32).min)


class LightweightConvolution(nn.Layer):
    """Lightweight Convolution layer.

    This implementation is based on
    https://github.com/pytorch/fairseq/tree/master/fairseq

    Args:
        wshare (int): 
            the number of kernel of convolution
        n_feat (int): 
            the number of features
        dropout_rate (float): 
            dropout_rate
        kernel_size (int): 
            kernel size (length)
        use_kernel_mask (bool): 
            Use causal mask or not for convolution kernel
        use_bias (bool): 
            Use bias term or not.

    """
    def __init__(
        self,
        wshare,
        n_feat,
        dropout_rate,
        kernel_size,
        use_kernel_mask=False,
        use_bias=False,
    ):
        """Construct Lightweight Convolution layer."""
        super().__init__()

        assert n_feat % wshare == 0
        self.wshare = wshare
        self.use_kernel_mask = use_kernel_mask
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.padding_size = int(kernel_size / 2)

        # linear -> GLU -> lightconv -> linear
        self.linear1 = nn.Linear(n_feat, n_feat * 2)
        self.linear2 = nn.Linear(n_feat, n_feat)
        self.act = get_activation("glu")

        # lightconv related
        self.uniform_ = nn.initializer.Uniform()
        self.weight = paddle.to_tensor(numpy.random.uniform(
            0, 1, size=[self.wshare, 1, kernel_size]),
                                       dtype="float32")
        self.uniform_(self.weight)
        self.weight = paddle.create_parameter(
            shape=self.weight.shape,
            dtype=str(self.weight.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(self.weight))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = paddle.Tensor(n_feat)
            self.bias = paddle.create_parameter(
                shape=self.bias.shape,
                dtype=str(self.bias.numpy().dtype),
                default_initializer=paddle.nn.initializer.Assign(self.bias))

        # mask of kernel
        kernel_mask0 = paddle.zeros([self.wshare, int(kernel_size / 2)])
        kernel_mask1 = paddle.ones([self.wshare, int(kernel_size / 2 + 1)])
        self.kernel_mask = paddle.concat((kernel_mask1, kernel_mask0),
                                         axis=-1).unsqueeze(1)

    def forward(self, query, key, value, mask):
        """Forward of 'Lightweight Convolution'.

        This function takes query, key and value but uses only query.
        This is just for compatibility with self-attention layer (attention.py)

        Args:
            query (Tensor): 
                input tensor. (batch, time1, d_model)
            key (Tensor): 
                NOT USED. (batch, time2, d_model)  
            value (Tensor): 
                NOT USED. (batch, time2, d_model) 
            mask : (Tensor):
                (batch, time1, time2) mask

        Return:
            Tensor: ouput. (batch, time1, d_model) 

        """
        # linear -> GLU -> lightconv -> linear
        x = query
        B, T, C = x.shape
        H = self.wshare

        # first liner layer
        x = self.linear1(x)

        # GLU activation
        x = self.act(x)

        # lightconv
        # B x C x T
        x = x.transpose([0, 2, 1]).reshape([-1, H, T])
        weight = F.dropout(self.weight,
                           self.dropout_rate,
                           training=self.training)
        if self.use_kernel_mask:
            weight = masked_fill(weight, self.kernel_mask == 0.0, float("-inf"))
            # weight = weight.masked_fill(self.kernel_mask == 0.0, float("-inf"))
        weight = F.softmax(weight, axis=-1)
        x = F.conv1d(x, weight, padding=self.padding_size,
                     groups=self.wshare).reshape([B, C, T])
        if self.use_bias:
            x = x + self.bias.reshape([1, -1, 1])
        # B x T x C
        x = x.transpose([0, 2, 1])

        if mask is not None and not self.use_kernel_mask:
            mask = mask.transpose([0, 2, 1])
            # x = x.masked_fill(mask == 0, 0.0)
            x = masked_fill(x, mask == 0, 0.0)

        # second linear layer
        x = self.linear2(x)
        return x
