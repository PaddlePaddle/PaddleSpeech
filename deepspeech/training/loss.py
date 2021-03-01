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

import logging

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

logger = logging.getLogger(__name__)

__all__ = ['CTCLoss']


def ctc_loss(logits,
             labels,
             input_lengths,
             label_lengths,
             blank=0,
             reduction='mean',
             norm_by_times=True):
    #logger.info("my ctc loss with norm by times")
    ## https://github.com/PaddlePaddle/Paddle/blob/f5ca2db2cc/paddle/fluid/operators/warpctc_op.h#L403
    loss_out = paddle.fluid.layers.warpctc(logits, labels, blank, norm_by_times,
                                           input_lengths, label_lengths)

    loss_out = paddle.fluid.layers.squeeze(loss_out, [-1])
    logger.info(f"warpctc loss: {loss_out}/{loss_out.shape} ")
    assert reduction in ['mean', 'sum', 'none']
    if reduction == 'mean':
        loss_out = paddle.mean(loss_out / label_lengths)
    elif reduction == 'sum':
        loss_out = paddle.sum(loss_out)
    logger.info(f"ctc loss: {loss_out}")
    return loss_out


F.ctc_loss = ctc_loss


class CTCLoss(nn.Layer):
    def __init__(self, blank_id):
        super().__init__()
        # last token id as blank id
        self.loss = nn.CTCLoss(blank=blank_id, reduction='sum')

    def forward(self, logits, text, logits_len, text_len):
        # warp-ctc do softmax on activations
        # warp-ctc need activation with shape [T, B, V + 1]
        logits = logits.transpose([1, 0, 2])

        ctc_loss = self.loss(logits, text, logits_len, text_len)
        return ctc_loss
