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
# This is modified from SpeechBrain
# https://github.com/speechbrain/speechbrain/blob/085be635c07f16d42cd1295045bc46c407f1e15b/speechbrain/nnet/losses.py
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class AngularMargin(nn.Layer):
    def __init__(self, margin=0.0, scale=1.0):
        """An implementation of Angular Margin (AM) proposed in the following
           paper: '''Margin Matters: Towards More Discriminative Deep Neural Network
           Embeddings for Speaker Recognition''' (https://arxiv.org/abs/1906.07317)

        Args:
            margin (float, optional): The margin for cosine similiarity. Defaults to 0.0.
            scale (float, optional): The scale for cosine similiarity. Defaults to 1.0.
        """
        super(AngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets):
        outputs = outputs - self.margin * targets
        return self.scale * outputs


class AdditiveAngularMargin(AngularMargin):
    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        """The Implementation of Additive Angular Margin (AAM) proposed
       in the following paper: '''Margin Matters: Towards More Discriminative Deep Neural Network Embeddings for Speaker Recognition'''
       (https://arxiv.org/abs/1906.07317)

        Args:
            margin (float, optional): margin factor. Defaults to 0.0.
            scale (float, optional): scale factor. Defaults to 1.0.
            easy_margin (bool, optional): easy_margin flag. Defaults to False.
        """
        super(AdditiveAngularMargin, self).__init__(margin, scale)
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        cosine = outputs.astype('float32')
        sine = paddle.sqrt(1.0 - paddle.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            phi = paddle.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs


class LogSoftmaxWrapper(nn.Layer):
    def __init__(self, loss_fn):
        """Speaker identificatin loss function wrapper 
           including all of compositions of the loss transformation
        Args:
            loss_fn (_type_): the loss value of a batch
        """
        super(LogSoftmaxWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.criterion = paddle.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, length=None):
        targets = F.one_hot(targets, outputs.shape[1])
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, axis=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss
