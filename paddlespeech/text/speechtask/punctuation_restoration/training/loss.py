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
import paddle.nn.functional as F


class FocalLossHX(nn.Layer):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        # print('input')
        # print(input.shape)
        # print(target.shape)

        if input.dim() > 2:
            input = paddle.reshape(
                input,
                shape=[input.size(0), input.size(1), -1])  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = paddle.reshape(
                input, shape=[-1, input.size(2)])  # N,H*W,C => N*H*W,C
        target = paddle.reshape(target, shape=[-1])

        logpt = F.log_softmax(input)
        # print('logpt')
        # print(logpt.shape)
        # print(logpt)

        # get true class column from each row
        all_rows = paddle.arange(len(input))
        # print(target)
        log_pt = logpt.numpy()[all_rows.numpy(), target.numpy()]

        pt = paddle.to_tensor(log_pt, dtype='float64').exp()
        ce = F.cross_entropy(input, target, reduction='none')
        # print('ce')
        # print(ce.shape)

        loss = (1 - pt)**self.gamma * ce
        # print('ce:%f'%ce.mean())
        # print('fl:%f'%loss.mean())
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalLoss(nn.Layer):
    """
    Focal Loss.
    Code referenced from:
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    Args:
        gamma (float): the coefficient of Focal Loss.
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logit, label):
        #####logit = F.softmax(logit)
        # logit = paddle.reshape(
        #     logit, [logit.shape[0], logit.shape[1], -1])  # N,C,H,W => N,C,H*W
        # logit = paddle.transpose(logit, [0, 2, 1])  # N,C,H*W => N,H*W,C
        # logit = paddle.reshape(logit,
        #                        [-1, logit.shape[2]])  # N,H*W,C => N*H*W,C
        label = paddle.reshape(label, [-1, 1])
        range_ = paddle.arange(0, label.shape[0])
        range_ = paddle.unsqueeze(range_, axis=-1)
        label = paddle.cast(label, dtype='int64')
        label = paddle.concat([range_, label], axis=-1)
        logpt = F.log_softmax(logit)
        logpt = paddle.gather_nd(logpt, label)

        pt = paddle.exp(logpt.detach())
        loss = -1 * (1 - pt)**self.gamma * logpt
        loss = paddle.mean(loss)
        # print(loss)
        # print(logpt)
        return loss
