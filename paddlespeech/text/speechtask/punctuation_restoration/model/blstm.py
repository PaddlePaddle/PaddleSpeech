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
import paddle
import paddle.nn as nn
import paddle.nn.initializer as I


class BiLSTM(nn.Layer):
    """LSTM for Punctuation Restoration
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 num_class,
                 init_scale=0.1):
        super(BiLSTM, self).__init__()
        # hyper parameters
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_class = num_class

        # 网络中的层
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))
        # print(hidden_size)
        # print(embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction="bidirect",
            weight_ih_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            weight_hh_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))
        # Here is a one direction LSTM. If bidirection LSTM, (hidden_size*2(,))
        self.fc = nn.Linear(
            in_features=hidden_size * 2,
            out_features=num_class,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            bias_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))
        # self.fc = nn.Linear(hidden_size, num_class)

        self.softmax = nn.Softmax()

    def forward(self, input):
        """The forward process of Net
        Parameters
        ----------
        inputs : tensor
            Training data, batch first
        """
        # Inherit the knowledge of context

        # hidden = self.init_hidden(inputs.size(0))
        # print('input_size',inputs.size())
        embedding = self.embedding(input)
        # print('embedding_size', embedding.size())
        # packed = pack_sequence(embedding, inputs_lengths, batch_first=True)
        # embedding本身是同样长度的，用这个函数主要是为了用pack
        # *****************************************************************************
        y, (_, _) = self.lstm(embedding)

        # print(y.size())
        y = self.fc(y)
        y = paddle.reshape(y, shape=[-1, self.num_class])
        logit = self.softmax(y)
        return y, logit
