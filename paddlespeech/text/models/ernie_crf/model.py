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
from paddlenlp.layers.crf import LinearChainCrf
from paddlenlp.layers.crf import LinearChainCrfLoss
from paddlenlp.layers.crf import ViterbiDecoder
from paddlenlp.transformers import ErnieForTokenClassification


class ErnieCrf(nn.Layer):
    def __init__(self,
                 num_classes,
                 pretrained_token='ernie-1.0',
                 crf_lr=100,
                 **kwargs):
        super().__init__()
        self.ernie = ErnieForTokenClassification.from_pretrained(
            pretrained_token, num_labels=num_classes, **kwargs)
        self.num_classes = num_classes
        self.crf = LinearChainCrf(self.num_classes,
                                  crf_lr=crf_lr,
                                  with_start_stop_tag=False)
        self.crf_loss = LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions,
                                              with_start_stop_tag=False)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                lengths=None,
                labels=None):
        logits = self.ernie(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids)

        if lengths is None:
            lengths = paddle.ones(shape=[input_ids.shape[0]],
                                  dtype=paddle.int64) * input_ids.shape[1]

        _, prediction = self.viterbi_decoder(logits, lengths)
        prediction = prediction.reshape([-1])

        if labels is not None:
            labels = labels.reshape([input_ids.shape[0], -1])
            loss = self.crf_loss(logits, lengths, labels)
            avg_loss = paddle.mean(loss)
            return avg_loss, prediction
        else:
            return prediction
