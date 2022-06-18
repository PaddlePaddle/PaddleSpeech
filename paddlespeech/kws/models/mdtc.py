# Copyright (c) 2021 Jingyong Hou (houjingyong@gmail.com)
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
# Modified from wekws(https://github.com/wenet-e2e/wekws)
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DSDilatedConv1d(nn.Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int=1,
            stride: int=1,
            bias: bool=True, ):
        super(DSDilatedConv1d, self).__init__()
        self.receptive_fields = dilation * (kernel_size - 1)
        self.conv = nn.Conv1D(
            in_channels,
            in_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            stride=stride,
            groups=in_channels,
            bias_attr=bias, )
        self.bn = nn.BatchNorm1D(in_channels)
        self.pointwise = nn.Conv1D(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            dilation=1,
            bias_attr=bias)

    def forward(self, inputs: paddle.Tensor):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.pointwise(outputs)
        return outputs


class TCNBlock(nn.Layer):
    def __init__(
            self,
            in_channels: int,
            res_channels: int,
            kernel_size: int,
            dilation: int,
            causal: bool, ):
        super(TCNBlock, self).__init__()
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.receptive_fields = dilation * (kernel_size - 1)
        self.half_receptive_fields = self.receptive_fields // 2
        self.conv1 = DSDilatedConv1d(
            in_channels=in_channels,
            out_channels=res_channels,
            kernel_size=kernel_size,
            dilation=dilation, )
        self.bn1 = nn.BatchNorm1D(res_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1D(
            in_channels=res_channels, out_channels=res_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1D(res_channels)
        self.relu2 = nn.ReLU()

    def forward(self, inputs: paddle.Tensor):
        outputs = self.relu1(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.causal:
            inputs = inputs[:, :, self.receptive_fields:]
        else:
            inputs = inputs[:, :, self.half_receptive_fields:
                            -self.half_receptive_fields]
        if self.in_channels == self.res_channels:
            res_out = self.relu2(outputs + inputs)
        else:
            res_out = self.relu2(outputs)
        return res_out


class TCNStack(nn.Layer):
    def __init__(
            self,
            in_channels: int,
            stack_num: int,
            stack_size: int,
            res_channels: int,
            kernel_size: int,
            causal: bool, ):
        super(TCNStack, self).__init__()
        self.in_channels = in_channels
        self.stack_num = stack_num
        self.stack_size = stack_size
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.res_blocks = self.stack_tcn_blocks()
        self.receptive_fields = self.calculate_receptive_fields()
        self.res_blocks = nn.Sequential(*self.res_blocks)

    def calculate_receptive_fields(self):
        receptive_fields = 0
        for block in self.res_blocks:
            receptive_fields += block.receptive_fields
        return receptive_fields

    def build_dilations(self):
        dilations = []
        for s in range(0, self.stack_size):
            for l in range(0, self.stack_num):
                dilations.append(2**l)
        return dilations

    def stack_tcn_blocks(self):
        dilations = self.build_dilations()
        res_blocks = nn.LayerList()

        res_blocks.append(
            TCNBlock(
                self.in_channels,
                self.res_channels,
                self.kernel_size,
                dilations[0],
                self.causal, ))
        for dilation in dilations[1:]:
            res_blocks.append(
                TCNBlock(
                    self.res_channels,
                    self.res_channels,
                    self.kernel_size,
                    dilation,
                    self.causal, ))
        return res_blocks

    def forward(self, inputs: paddle.Tensor):
        outputs = self.res_blocks(inputs)
        return outputs


class MDTC(nn.Layer):
    def __init__(
            self,
            stack_num: int,
            stack_size: int,
            in_channels: int,
            res_channels: int,
            kernel_size: int,
            causal: bool=True, ):
        super(MDTC, self).__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.causal = causal
        self.preprocessor = TCNBlock(
            in_channels, res_channels, kernel_size, dilation=1, causal=causal)
        self.relu = nn.ReLU()
        self.blocks = nn.LayerList()
        self.receptive_fields = self.preprocessor.receptive_fields
        for i in range(stack_num):
            self.blocks.append(
                TCNStack(res_channels, stack_size, 1, res_channels, kernel_size,
                         causal))
            self.receptive_fields += self.blocks[-1].receptive_fields
        self.half_receptive_fields = self.receptive_fields // 2
        self.hidden_dim = res_channels

    def forward(self, x: paddle.Tensor):
        if self.causal:
            outputs = F.pad(x, (0, 0, self.receptive_fields, 0, 0, 0),
                            'constant')
        else:
            outputs = F.pad(
                x,
                (0, 0, self.half_receptive_fields, self.half_receptive_fields,
                 0, 0),
                'constant', )
        outputs = outputs.transpose([0, 2, 1])
        outputs_list = []
        outputs = self.relu(self.preprocessor(outputs))
        for block in self.blocks:
            outputs = block(outputs)
            outputs_list.append(outputs)

        normalized_outputs = []
        output_size = outputs_list[-1].shape[-1]
        for x in outputs_list:
            remove_length = x.shape[-1] - output_size
            if self.causal and remove_length > 0:
                normalized_outputs.append(x[:, :, remove_length:])
            elif not self.causal and remove_length > 1:
                half_remove_length = remove_length // 2
                normalized_outputs.append(
                    x[:, :, half_remove_length:-half_remove_length])
            else:
                normalized_outputs.append(x)

        outputs = paddle.zeros_like(
            outputs_list[-1], dtype=outputs_list[-1].dtype)
        for x in normalized_outputs:
            outputs += x
        outputs = outputs.transpose([0, 2, 1])
        return outputs, None


class KWSModel(nn.Layer):
    def __init__(self, backbone, num_keywords):
        super(KWSModel, self).__init__()
        self.backbone = backbone
        self.linear = nn.Linear(self.backbone.hidden_dim, num_keywords)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        outputs = self.backbone(x)
        outputs = self.linear(outputs)
        return self.activation(outputs)
