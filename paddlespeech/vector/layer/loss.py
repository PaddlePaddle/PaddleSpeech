# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


"""
customize loss functions
"""
import math
import numpy as np
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class FocalLoss(nn.Layer):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
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
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                      reduction="none")

    def forward(self, inputs, targets):
        """
        Forword inference.

        Args:
            x: input tensor
            target: target label tensor
        """
        ce_loss = self.ce(inputs, targets)
        pt = paddle.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

    @staticmethod
    def add_specific_args(parent_parser):
        """
        Static class method for Command arguments configuration

        Args:
            parent_parser: instance of argparse.ArgumentParser

        Returns:
            parsers: instance of argparse.ArgumentParser
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--focal_alpha", action="store", type=float, default=1.0,
                            help="focal_alpha(float): The scale value of focal loss, Default: 1.0")
        parser.add_argument("--focal_gamma", action="store", type=float, default=0,
                            help="gamma(float): The gamma value of amsoftmax, Default: 0")

        return parser


class AMSoftmax(nn.Layer):
    """
    AMSoftmax loss funtion
    """
    def __init__(self,
                 m=0.3,
                 s=15.0,
                 gamma=0):
        super(AMSoftmax, self).__init__()
        self.gamma = gamma
        self._m = m
        self._s = s
        self.ce = paddle.nn.CrossEntropyLoss(reduction="none")

    def forward(self, cos_theta, target):
        """
            Forward inference

            Args:
                x: input tensor
                target: target label tensor
        """
        assert cos_theta.shape[0] == target.shape[0]
        n_classes = cos_theta.shape[1]

        target = paddle.reshape(target, (-1, 1))    # size=(B,1)
        one_hot = paddle.fluid.layers.one_hot(target, n_classes)
        cos_theta_m = cos_theta - self._m * one_hot
        cos_theta_sm = self._s * cos_theta_m

        logit = self.ce(cos_theta_sm, target)
        pt = paddle.exp(-logit)
        loss = (1 - pt) ** self.gamma * logit
        loss = loss.mean()

        return loss

    @staticmethod
    def add_specific_args(parent_parser):
        """
        Static class method for Command arguments configuration

        Args:
            parent_parser: instance of argparse.ArgumentParser

        Returns:
            parsers: instance of argparse.ArgumentParser
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--amsoftmax_s", action="store", type=float, default=15,
                            help="s(float): The scale value of amsoftmax, Default: 15")
        parser.add_argument("--amsoftmax_m", action="store", type=float, default=0.3,
                            help="m(float): The margin value of amsoftmax, Default: 0.3")
        parser.add_argument("--amsoftmax_gamma", action="store", type=float, default=0,
                            help="gamma(float): The gamma value of amsoftmax, Default: 0")

        return parser


class AAMSoftmax(nn.Layer):
    """
    AAMSoftmax loss funtion
    """
    def __init__(self,
                 m=0.2,
                 s=20.0):
        super(AAMSoftmax, self).__init__()
        self._m = m
        self._s = s
        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.th = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m
        self.ce = paddle.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, cos_theta, target):
        """
            Forward inference

            Args:
                x: input tensor
                target: target label tensor
        """
        assert cos_theta.shape[0] == target.shape[0]
        n_classes = cos_theta.shape[1]

        target = paddle.reshape(target, (-1, 1))    # size=(B,1)
        one_hot = paddle.fluid.layers.one_hot(target, n_classes)

        sin_theta = paddle.sqrt(1.0 - paddle.pow(cos_theta, 2))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m  # cos(theta + m)
        phi = paddle.where(cos_theta > self.th, phi, cos_theta - self.mm)
        outputs = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)

        loss = self.ce(self._s * outputs, target)

        return loss

    @staticmethod
    def add_specific_args(parent_parser):
        """
        Static class method for Command arguments configuration
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--aamsoftmax_s", action="store", type=float, default=20,
                            help="s(float): The scale value of aamsoftmax, Default: 20")
        parser.add_argument("--aamsoftmax_m", action="store", type=float, default=0.2,
                            help="m(float): The margin value of aamsoftmax, Default: 0.2")

        return parser


class NCELoss(nn.Layer):
    """
    Noise Contrastive Estimation loss funtion

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
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import paddle.nn as nn
            ...

            model = Trainer(Xvector(23, 4609))
            scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=args.lr, ... )
            optim = paddle.optimizer.SGD(learning_rate=args.lr, ... )
            callbacks = [ ProgBarLogger(args.log_freq, verbose=args.verbose), ... ]

            Q = Q_from_tokens(4609)
            model.prepare(optim, NCELoss(Q), Accuracy(topk=(1, 5)))

            model.fit(train_dataset,val_dataset, ... )

    """
    def __init__(self,
                 Q,
                 noise_ratio=100,
                 Z_offset=9.5):
        super(NCELoss, self).__init__()
        assert type(noise_ratio) is int
        # the Q is prior model, uniform or guassian
        # the N is the number of vocabulary/speakers
        # the K is the noise sampling times
        self.Q = paddle.to_tensor(Q, stop_gradient = False)
        self.N = self.Q.shape[0]
        self.K = noise_ratio
        # exp(X)/Z = exp(X-Z_offset)
        self.Z_offset = Z_offset

    def update_Z_offset(self, new_Z):
        """
            update Z offset
        """
        self.Z_offset = math.log(new_Z)

    def forward(self, output, target):
        """
            Forward inference

            Args:
                output: input tensor, which is the NN output
                target: target label tensor
        """
        output = paddle.reshape(output, [-1, self.N])
        B = output.shape[0]
        noise_idx = self.get_noise(B)
        idx = self.get_combined_idx(target, noise_idx)
        P_target, P_noise = self.get_prob(idx, output, sep_target = True)
        Q_target, Q_noise = self.get_Q(idx)
        loss = self.nce_loss(P_target, P_noise, Q_noise, Q_target)
        return loss.mean()

    def get_Q(self, idx, sep_target=True):
        """
            get prior model of batchsize data
        """
        idx_size = idx.size
        prob_model = paddle.to_tensor(self.Q.numpy()[paddle.reshape(idx, [-1]).numpy()])
        prob_model = paddle.reshape(prob_model, [idx.shape[0], idx.shape[1]])
        if sep_target:
            #target, noise
            return prob_model[:, 0], prob_model[:, 1:]
        else:
            return prob_model

    def get_prob(self, idx, scores, sep_target=True):
        """
            post processing the score of post model(output of nn) of batchsize data
        """
        scores = self.get_scores(idx, scores)
        scale = paddle.to_tensor([self.Z_offset], dtype = 'float32')
        scores = paddle.add(scores, -scale)
        prob = paddle.exp(scores)
        if sep_target:
            #target, noise
            return prob[:, 0], prob[:, 1:]
        else:
            return prob

    def get_scores(self, idx, scores):
        """
            get the score of post model(output of nn) of batchsize data
        """
        B, N = scores.shape
        #the K = self.K + 1; K->noise 1->target
        K = idx.shape[1]
        idx_increment = paddle.to_tensor(N * paddle.reshape(paddle.arange(B), [B, 1]) * paddle.ones([1, K]),
                                         dtype = "int64", stop_gradient = False)
        new_idx = idx_increment + idx
        new_scores = paddle.index_select(paddle.reshape(scores, [-1]), paddle.reshape(new_idx, [-1]))

        return paddle.reshape(new_scores, [B, K])

    def get_noise(self, batch_size, uniform=True):
        """
            select noise sample
        """
        # this function would also convert the target into (-1, N)
        if uniform:
            noise = np.random.randint(self.N, size = self.K * batch_size)
        else:
            noise = np.random.choice(self.N, self.K * batch_size, replace = True, p = self.Q.data )
        noise = paddle.to_tensor(noise, dtype = 'int64', stop_gradient = False)
        noise_idx = paddle.reshape(noise, [batch_size, self.K])
        return noise_idx

    def get_combined_idx(self, target_idx, noise_idx):
        """
            combined target and noise
        """
        target_idx = paddle.reshape(target_idx, [-1, 1])
        return paddle.concat((target_idx, noise_idx), 1)

    def nce_loss(self, prob_model, prob_noise_in_model, prob_noise, prob_target_in_noise):
        """
            combined the loss of target and noise
        """
        # prob_model: P_target
        # prob_target_in_noise: Q_target
        # prob_noise_in_model: P_noise
        # prob_noise: Q_noise
        def safe_log(tensor):
            """
            safe log
            """
            EPSILON = 1e-10
            return paddle.log(EPSILON + tensor)

        model_loss = safe_log(prob_model / (
            prob_model + self.K * prob_target_in_noise
        ))
        model_loss = paddle.reshape(model_loss, [-1])

        noise_loss = paddle.sum(
            safe_log((self.K * prob_noise) / (prob_noise_in_model + self.K * prob_noise)), -1
        )
        noise_loss = paddle.reshape(noise_loss, [-1])

        loss = - (model_loss + noise_loss)

        return loss
