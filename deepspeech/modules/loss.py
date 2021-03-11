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


# TODO(Hui Zhang): remove this hack, when `norm_by_times=True` is added
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


# TODO(Hui Zhang): remove this hack
F.ctc_loss = ctc_loss


class CTCLoss(nn.Layer):
    def __init__(self, blank=0, reduction='sum'):
        super().__init__()
        # last token id as blank id
        self.loss = nn.CTCLoss(blank=blank, reduction=reduction)

    def forward(self, logits, ys_pad, hlens, ys_lens):
        """Compute CTC loss.

        Args:
            logits ([paddle.Tensor]): [description]
            ys_pad ([paddle.Tensor]): [description]
            hlens ([paddle.Tensor]): [description]
            ys_lens ([paddle.Tensor]): [description]

        Returns:
            [paddle.Tensor]: scalar. If reduction is 'none', then (N), where N = \text{batch size}.
        """
        # warp-ctc need logits, and do softmax on logits by itself
        # warp-ctc need activation with shape [T, B, V + 1]
        # logits: (B, L, D) -> (L, B, D)
        logits = logits.transpose([1, 0, 2])
        loss = self.loss(logits, ys_pad, hlens, ys_lens)

        # wenet do batch-size average, deepspeech2 not do this
        # Batch-size average
        # loss = loss / paddle.shape(logits)[1]
        return loss
