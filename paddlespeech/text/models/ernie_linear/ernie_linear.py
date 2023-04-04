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

import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErnieForTokenClassification


class ErnieLinear(nn.Layer):
    def __init__(self,
                 num_classes=None,
                 pretrained_token='ernie-1.0',
                 cfg_path=None,
                 ckpt_path=None,
                 **kwargs):
        super(ErnieLinear, self).__init__()

        if cfg_path is not None and ckpt_path is not None:
            cfg_path = os.path.abspath(os.path.expanduser(cfg_path))
            ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path))

            assert os.path.isfile(
                cfg_path), 'Config file is not valid: {}'.format(cfg_path)
            assert os.path.isfile(
                ckpt_path), 'Checkpoint file is not valid: {}'.format(ckpt_path)

            self.ernie = ErnieForTokenClassification.from_pretrained(
                os.path.dirname(cfg_path))
        else:
            assert isinstance(
                num_classes, int
            ) and num_classes > 0, 'Argument `num_classes` must be an integer.'
            self.ernie = ErnieForTokenClassification.from_pretrained(
                pretrained_token, num_labels=num_classes, **kwargs)

        self.num_classes = self.ernie.num_labels
        self.softmax = nn.Softmax()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        y = self.ernie(input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       position_ids=position_ids)

        y = paddle.reshape(y, shape=[-1, self.num_classes])
        logits = self.softmax(y)

        return y, logits
