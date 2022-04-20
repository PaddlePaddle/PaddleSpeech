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


class NCELoss(nn.Layer):
    """Noise Contrastive Estimation loss funtion

    Noise Contrastive Estimation (NCE) is an approximation method that is used to
    work around the huge computational cost of large softmax layer.
    The basic idea is to convert the prediction problem into classification problem
    at training stage. It has been proved that these two criterions converges to
    the same minimal point as long as noise distribution is close enough to real one.

    NCE bridges the gap between generative models and discriminative models,
    rather than simply speedup the softmax layer.
    With NCE, you can turn almost anything into posterior with less effort (I think).

    Refs:
    NCE：http://www.cs.helsinki.fi/u/ahyvarin/papers/Gutmann10AISTATS.pdf
    Thanks: https://github.com/mingen-pan/easy-to-use-NCE-RNN-for-Pytorch/blob/master/nce.py

    Examples:
    Q = Q_from_tokens(output_dim)
    NCELoss(Q)
    """

    def __init__(self, Q, noise_ratio=100, Z_offset=9.5):
        """Noise Contrastive Estimation loss funtion

        Args:
            Q (tensor): prior model, uniform or guassian
            noise_ratio (int, optional): noise sampling times. Defaults to 100.
            Z_offset (float, optional): scale of post processing the score. Defaults to 9.5.
        """
        super(NCELoss, self).__init__()
        assert type(noise_ratio) is int
        self.Q = paddle.to_tensor(Q, stop_gradient=False)
        self.N = self.Q.shape[0]
        self.K = noise_ratio
        self.Z_offset = Z_offset

    def forward(self, output, target):
        """Forward inference

        Args:
            output (tensor): the model output, which is the input of loss function
        """
        output = paddle.reshape(output, [-1, self.N])
        B = output.shape[0]
        noise_idx = self.get_noise(B)
        idx = self.get_combined_idx(target, noise_idx)
        P_target, P_noise = self.get_prob(idx, output, sep_target=True)
        Q_target, Q_noise = self.get_Q(idx)
        loss = self.nce_loss(P_target, P_noise, Q_noise, Q_target)
        return loss.mean()

    def get_Q(self, idx, sep_target=True):
        """Get prior model of batchsize data
        """
        idx_size = idx.size
        prob_model = paddle.to_tensor(
            self.Q.numpy()[paddle.reshape(idx, [-1]).numpy()])
        prob_model = paddle.reshape(prob_model, [idx.shape[0], idx.shape[1]])
        if sep_target:
            return prob_model[:, 0], prob_model[:, 1:]
        else:
            return prob_model

    def get_prob(self, idx, scores, sep_target=True):
        """Post processing the score of post model(output of nn) of batchsize data
        """
        scores = self.get_scores(idx, scores)
        scale = paddle.to_tensor([self.Z_offset], dtype='float64')
        scores = paddle.add(scores, -scale)
        prob = paddle.exp(scores)
        if sep_target:
            return prob[:, 0], prob[:, 1:]
        else:
            return prob

    def get_scores(self, idx, scores):
        """Get the score of post model(output of nn) of batchsize data
        """
        B, N = scores.shape
        K = idx.shape[1]
        idx_increment = paddle.to_tensor(
            N * paddle.reshape(paddle.arange(B), [B, 1]) * paddle.ones([1, K]),
            dtype="int64",
            stop_gradient=False)
        new_idx = idx_increment + idx
        new_scores = paddle.index_select(
            paddle.reshape(scores, [-1]), paddle.reshape(new_idx, [-1]))

        return paddle.reshape(new_scores, [B, K])

    def get_noise(self, batch_size, uniform=True):
        """Select noise sample
        """
        if uniform:
            noise = np.random.randint(self.N, size=self.K * batch_size)
        else:
            noise = np.random.choice(
                self.N, self.K * batch_size, replace=True, p=self.Q.data)
        noise = paddle.to_tensor(noise, dtype='int64', stop_gradient=False)
        noise_idx = paddle.reshape(noise, [batch_size, self.K])
        return noise_idx

    def get_combined_idx(self, target_idx, noise_idx):
        """Combined target and noise
        """
        target_idx = paddle.reshape(target_idx, [-1, 1])
        return paddle.concat((target_idx, noise_idx), 1)

    def nce_loss(self, prob_model, prob_noise_in_model, prob_noise,
                 prob_target_in_noise):
        """Combined the loss of target and noise
        """

        def safe_log(tensor):
            """Safe log
            """
            EPSILON = 1e-10
            return paddle.log(EPSILON + tensor)

        model_loss = safe_log(prob_model /
                              (prob_model + self.K * prob_target_in_noise))
        model_loss = paddle.reshape(model_loss, [-1])

        noise_loss = paddle.sum(
            safe_log((self.K * prob_noise) /
                     (prob_noise_in_model + self.K * prob_noise)), -1)
        noise_loss = paddle.reshape(noise_loss, [-1])

        loss = -(model_loss + noise_loss)

        return loss


class FocalLoss(nn.Layer):
    """This criterion is a implemenation of Focal Loss, which is proposed in 
    Focal Loss for Dense Object Detection.

        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    The losses are averaged across observations for each minibatch.

    Args:
        alpha(1D Tensor, Variable) : the scalar factor for this criterion
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                putting more focus on hard, misclassiﬁed examples
        size_average(bool): By default, the losses are averaged over observations for each minibatch.
                            However, if the field size_average is set to False, the losses are
                            instead summed for each minibatch.
    """

    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none")

    def forward(self, outputs, targets):
        """Forword inference.

        Args:
            outputs: input tensor
            target: target label tensor
        """
        ce_loss = self.ce(outputs, targets)
        pt = paddle.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


if __name__ == "__main__":
    import numpy as np
    from paddlespeech.vector.utils.vector_utils import Q_from_tokens
    paddle.set_device("cpu")

    input_data = paddle.uniform([5, 100], dtype="float64")
    label_data = np.random.randint(0, 100, size=(5)).astype(np.int64)

    input = paddle.to_tensor(input_data)
    label = paddle.to_tensor(label_data)

    loss1 = FocalLoss()
    loss = loss1.forward(input, label)
    print("loss: %.5f" % (loss))

    Q = Q_from_tokens(100)
    loss2 = NCELoss(Q)
    loss = loss2.forward(input, label)
    print("loss: %.5f" % (loss))
