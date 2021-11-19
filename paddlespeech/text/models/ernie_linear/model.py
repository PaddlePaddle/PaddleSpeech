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
from paddlenlp.transformers import ErnieForTokenClassification


class ErnieLinear(nn.Layer):
    def __init__(self, num_classes, pretrained_token='ernie-1.0', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.ernie = ErnieForTokenClassification.from_pretrained(
            pretrained_token, num_classes=num_classes, **kwargs)
        self.softmax = nn.Softmax()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        y = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids)

        y = paddle.reshape(y, shape=[-1, self.num_classes])
        logit = self.softmax(y)

        return y, logit
