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
import os

import paddle.nn as nn
import paddle.nn.functional as F

from paddleaudio.utils.download import load_state_dict_from_url
from paddleaudio.utils.env import MODEL_HOME

__all__ = ['CNN14', 'CNN10', 'CNN6', 'cnn14', 'cnn10', 'cnn6']

pretrained_model_urls = {
    'cnn14': 'https://bj.bcebos.com/paddleaudio/models/panns_cnn14.pdparams',
    'cnn10': 'https://bj.bcebos.com/paddleaudio/models/panns_cnn10.pdparams',
    'cnn6': 'https://bj.bcebos.com/paddleaudio/models/panns_cnn6.pdparams',
}


class ConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias_attr=False)
        self.conv2 = nn.Conv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.bn2 = nn.BatchNorm2D(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(
                x, kernel_size=pool_size) + F.max_pool2d(
                    x, kernel_size=pool_size)
        else:
            raise Exception(
                f'Pooling type of {pool_type} is not supported. It must be one of "max", "avg" and "avg+max".'
            )
        return x


class ConvBlock5x5(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(
                x, kernel_size=pool_size) + F.max_pool2d(
                    x, kernel_size=pool_size)
        else:
            raise Exception(
                f'Pooling type of {pool_type} is not supported. It must be one of "max", "avg" and "avg+max".'
            )
        return x


class CNN14(nn.Layer):
    """
    The CNN14(14-layer CNNs) mainly consist of 6 convolutional blocks while each convolutional
    block consists of 2 convolutional layers with a kernel size of 3 × 3.

    Reference:
        PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
        https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 2048

    def __init__(self, extract_embedding: bool=True):

        super(CNN14, self).__init__()
        self.bn0 = nn.BatchNorm2D(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

    def forward(self, x):
        x.stop_gradient = False
        x = x.transpose([0, 3, 2, 1])
        x = self.bn0(x)
        x = x.transpose([0, 3, 2, 1])

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.mean(axis=3)
        x = x.max(axis=2) + x.mean(axis=2)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))
        return output


class CNN10(nn.Layer):
    """
    The CNN10(14-layer CNNs) mainly consist of 4 convolutional blocks while each convolutional
    block consists of 2 convolutional layers with a kernel size of 3 × 3.

    Reference:
        PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
        https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 512

    def __init__(self, extract_embedding: bool=True):

        super(CNN10, self).__init__()
        self.bn0 = nn.BatchNorm2D(64)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

    def forward(self, x):
        x.stop_gradient = False
        x = x.transpose([0, 3, 2, 1])
        x = self.bn0(x)
        x = x.transpose([0, 3, 2, 1])

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.mean(axis=3)
        x = x.max(axis=2) + x.mean(axis=2)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))
        return output


class CNN6(nn.Layer):
    """
    The CNN14(14-layer CNNs) mainly consist of 4 convolutional blocks while each convolutional
    block consists of 1 convolutional layers with a kernel size of 5 × 5.

    Reference:
        PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
        https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 512

    def __init__(self, extract_embedding: bool=True):

        super(CNN6, self).__init__()
        self.bn0 = nn.BatchNorm2D(64)
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

    def forward(self, x):
        x.stop_gradient = False
        x = x.transpose([0, 3, 2, 1])
        x = self.bn0(x)
        x = x.transpose([0, 3, 2, 1])

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.mean(axis=3)
        x = x.max(axis=2) + x.mean(axis=2)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))
        return output


def cnn14(pretrained: bool=False, extract_embedding: bool=True) -> CNN14:
    model = CNN14(extract_embedding=extract_embedding)
    if pretrained:
        state_dict = load_state_dict_from_url(
            url=pretrained_model_urls['cnn14'],
            path=os.path.join(MODEL_HOME, 'panns'))
        model.set_state_dict(state_dict)
    return model


def cnn10(pretrained: bool=False, extract_embedding: bool=True) -> CNN10:
    model = CNN10(extract_embedding=extract_embedding)
    if pretrained:
        state_dict = load_state_dict_from_url(
            url=pretrained_model_urls['cnn10'],
            path=os.path.join(MODEL_HOME, 'panns'))
        model.set_state_dict(state_dict)
    return model


def cnn6(pretrained: bool=False, extract_embedding: bool=True) -> CNN6:
    model = CNN6(extract_embedding=extract_embedding)
    if pretrained:
        state_dict = load_state_dict_from_url(
            url=pretrained_model_urls['cnn6'],
            path=os.path.join(MODEL_HOME, 'panns'))
        model.set_state_dict(state_dict)
    return model
