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
import paddle
from paddle import nn
import math
"""
    To align the initializer between paddle and torch, 
    the API below are set defalut initializer with priority higger than global initializer.
"""
global_init_type = None


class LayerNorm(nn.LayerNorm):
    def __init__(self,
                 normalized_shape,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        if weight_attr is None:
            weight_attr = paddle.ParamAttr(
                initializer=nn.initializer.Constant(1.0))
        if bias_attr is None:
            bias_attr = paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.0))
        super(LayerNorm, self).__init__(normalized_shape, epsilon, weight_attr,
                                        bias_attr, name)


class BatchNorm1D(nn.BatchNorm1D):
    def __init__(self,
                 num_features,
                 momentum=0.9,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCL',
                 name=None):
        if weight_attr is None:
            weight_attr = paddle.ParamAttr(
                initializer=nn.initializer.Constant(1.0))
        if bias_attr is None:
            bias_attr = paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.0))
        super(BatchNorm1D,
              self).__init__(num_features, momentum, epsilon, weight_attr,
                             bias_attr, data_format, name)


class Embedding(nn.Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 sparse=False,
                 weight_attr=None,
                 name=None):
        if weight_attr is None:
            weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal())
        super(Embedding, self).__init__(num_embeddings, embedding_dim,
                                        padding_idx, sparse, weight_attr, name)


class Linear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        if weight_attr is None:
            if global_init_type == "kaiming_uniform":
                weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform(fan_in=None, negative_slope=math.sqrt(5), nonlinearity='leaky_relu'))
        if bias_attr is None:
            if global_init_type == "kaiming_uniform":
                bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform(fan_in=None, negative_slope=math.sqrt(5), nonlinearity='leaky_relu'))
        super(Linear, self).__init__(in_features, out_features, weight_attr,
                                     bias_attr, name)


class Conv1D(nn.Conv1D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCL'):
        if weight_attr is None:
            if global_init_type == "kaiming_uniform":
                weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform(fan_in=None, negative_slope=math.sqrt(5), nonlinearity='leaky_relu'))
        if bias_attr is None:
            if global_init_type == "kaiming_uniform":
                bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform(fan_in=None, negative_slope=math.sqrt(5), nonlinearity='leaky_relu'))
        super(Conv1D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, padding_mode, weight_attr, bias_attr, data_format)


class Conv2D(nn.Conv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW'):
        if weight_attr is None:
            if global_init_type == "kaiming_uniform":
                weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform(fan_in=None, negative_slope=math.sqrt(5), nonlinearity='leaky_relu'))
        if bias_attr is None:
            if global_init_type == "kaiming_uniform":
                bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform(fan_in=None, negative_slope=math.sqrt(5), nonlinearity='leaky_relu'))
        super(Conv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, padding_mode, weight_attr, bias_attr, data_format)
