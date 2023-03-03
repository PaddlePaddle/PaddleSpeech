# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
Implementation of model from:
Kum et al. - "Joint Detection and Classification of Singing Voice Melody Using
Convolutional Recurrent Neural Networks" (2019)
Link: https://www.semanticscholar.org/paper/Joint-Detection-and-Classification-of-Singing-Voice-Kum-Nam/60a2ad4c7db43bace75805054603747fcd062c0d
"""
import paddle
from paddle import nn


class JDCNet(nn.Layer):
    """
    Joint Detection and Classification Network model for singing voice melody.
    """

    def __init__(self,
                 num_class: int=722,
                 seq_len: int=31,
                 leaky_relu_slope: float=0.01):
        super().__init__()
        self.seq_len = seq_len
        self.num_class = num_class
        # input: (B, num_class, T, n_mels)
        self.conv_block = nn.Sequential(
            # output: (B, out_channels, T, n_mels)
            nn.Conv2D(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias_attr=False),
            nn.BatchNorm2D(num_features=64),
            nn.LeakyReLU(leaky_relu_slope),
            # out: (B, out_channels, T, n_mels)
            nn.Conv2D(64, 64, 3, padding=1, bias_attr=False), )
        # output: (B, out_channels, T, n_mels//2)
        self.res_block1 = ResBlock(in_channels=64, out_channels=128)
        # output: (B, out_channels, T, n_mels//4) 
        self.res_block2 = ResBlock(in_channels=128, out_channels=192)
        # output: (B, out_channels, T, n_mels//8)  
        self.res_block3 = ResBlock(in_channels=192, out_channels=256)
        # pool block
        self.pool_block = nn.Sequential(
            nn.BatchNorm2D(num_features=256),
            nn.LeakyReLU(leaky_relu_slope),
            # (B, num_features, T, 2)
            nn.MaxPool2D(kernel_size=(1, 4)),
            nn.Dropout(p=0.5), )
        # input: (B, T, input_size) - resized from (B, input_size//2, T, 2)
        # output: (B, T, input_size)
        self.bilstm_classifier = nn.LSTM(
            input_size=512,
            hidden_size=256,
            time_major=False,
            direction='bidirectional')
        # input: (B * T, in_features)
        # output: (B * T, num_class)
        self.classifier = nn.Linear(
            in_features=512, out_features=self.num_class)

        # initialize weights
        self.apply(self.init_weights)

    def get_feature_GAN(self, x: paddle.Tensor):
        """Calculate feature_GAN.
        Args:
            x(Tensor(float32)): 
                Shape (B, num_class, n_mels, T).
        Returns:
            Tensor:
                Shape (B, num_features, n_mels//8, T).
        """
        x = x.astype(paddle.float32)
        x = x.transpose([0, 1, 3, 2] if len(x.shape) == 4 else [0, 2, 1])
        convblock_out = self.conv_block(x)
        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)
        GAN_feature = poolblock_out.transpose([0, 1, 3, 2] if len(
            poolblock_out.shape) == 4 else [0, 2, 1])
        return GAN_feature

    def forward(self, x: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            x(Tensor(float32)): 
                Shape (B, num_class, n_mels, seq_len).
        Returns:
            Tensor:
                classifier output consists of predicted pitch classes per frame.
                Shape: (B, seq_len, num_class).
            Tensor:
                GAN_feature. Shape: (B, num_features, n_mels//8, seq_len)
            Tensor:
                poolblock_out. Shape (B, seq_len, 512)
                
        """
        ###############################
        # forward pass for classifier #
        ###############################
        # (B, num_class, n_mels, T) -> (B, num_class, T, n_mels)
        x = x.transpose([0, 1, 3, 2] if len(x.shape) == 4 else
                        [0, 2, 1]).astype(paddle.float32)

        convblock_out = self.conv_block(x)
        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)
        GAN_feature = poolblock_out.transpose([0, 1, 3, 2] if len(
            poolblock_out.shape) == 4 else [0, 2, 1])
        poolblock_out = self.pool_block[2](poolblock_out)
        # (B, 256, seq_len, 2) => (B, seq_len, 256, 2) => (B, seq_len, 512)
        classifier_out = poolblock_out.transpose([0, 2, 1, 3]).reshape(
            (-1, self.seq_len, 512))
        self.bilstm_classifier.flatten_parameters()
        # ignore the hidden states
        classifier_out, _ = self.bilstm_classifier(classifier_out)
        # (B * seq_len, 512)
        classifier_out = classifier_out.reshape((-1, 512))
        classifier_out = self.classifier(classifier_out)
        # (B, seq_len, num_class)
        classifier_out = classifier_out.reshape(
            (-1, self.seq_len, self.num_class))
        return paddle.abs(classifier_out.squeeze()), GAN_feature, poolblock_out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.initializer.KaimingUniform()(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.Conv2D):
            nn.initializer.XavierNormal()(m.weight)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
            for p in m.parameters():
                if len(p.shape) >= 2:
                    nn.initializer.Orthogonal()(p)
                else:
                    nn.initializer.Normal()(p)


class ResBlock(nn.Layer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 leaky_relu_slope: float=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels
        # BN / LReLU / MaxPool layer before the conv layer - see Figure 1b in the paper
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2D(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope),
            # apply downsampling on the y axis only
            nn.MaxPool2D(kernel_size=(1, 2)), )

        # conv layers
        self.conv = nn.Sequential(
            nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        # 1 x 1 convolution layer to match the feature dimensions
        self.conv1by1 = None
        if self.downsample:
            self.conv1by1 = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias_attr=False)

    def forward(self, x: paddle.Tensor):
        """Calculate forward propagation.
        Args:
            x(Tensor(float32)): Shape (B, in_channels, T, n_mels).
        Returns:
            Tensor:
                The residual output, Shape (B, out_channels, T, n_mels//2).
        """
        x = self.pre_conv(x)
        if self.downsample:
            x = self.conv(x) + self.conv1by1(x)
        else:
            x = self.conv(x) + x
        return x
