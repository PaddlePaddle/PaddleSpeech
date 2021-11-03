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
from paddlenlp.transformers import BertForTokenClassification


class BertBLSTMPunc(nn.Layer):
    def __init__(self,
                 pretrained_token="bert-large-uncased",
                 output_size=4,
                 dropout=0.0,
                 bert_size=768,
                 blstm_size=128,
                 num_blstm_layers=2,
                 init_scale=0.1):
        super(BertBLSTMPunc, self).__init__()
        self.output_size = output_size
        self.bert = BertForTokenClassification.from_pretrained(
            pretrained_token, num_classes=bert_size)
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)

        self.lstm = nn.LSTM(
            input_size=bert_size,
            hidden_size=blstm_size,
            num_layers=num_blstm_layers,
            direction="bidirect",
            weight_ih_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)),
            weight_hh_attr=paddle.ParamAttr(initializer=I.Uniform(
                low=-init_scale, high=init_scale)))

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(blstm_size * 2, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # print('input :', x.shape)
        x = self.bert(x)  #[0]
        # print('after bert :', x.shape)

        y, (_, _) = self.lstm(x)
        # print('after lstm :', y.shape)
        y = self.fc(self.dropout(y))
        y = paddle.reshape(y, shape=[-1, self.output_size])
        # print('after fc :', y.shape)

        logit = self.softmax(y)
        # print('after softmax :', logit.shape)

        return y, logit


if __name__ == '__main__':
    print('start model')
    model = BertBLSTMPunc()
    x = paddle.randint(low=0, high=40, shape=[2, 5])
    print(x)
    y, logit = model(x)
