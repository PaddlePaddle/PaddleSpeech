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

logger = logging.getLogger(__name__)

__all__ = ['CTCLoss', "LabelSmoothingLoss"]


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


class LabelSmoothingLoss(nn.Layer):
    """Label-smoothing loss.
    In a standard CE loss, the label's data distribution is:
        [0,1,2] ->
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.
        e.g.
        smoothing=0.1
        [0,1,2] ->
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]

    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool=False):
        """Label-smoothing loss.

        Args:
            size (int): the number of class
            padding_idx (int): padding class id which will be ignored for loss
            smoothing (float): smoothing rate (0.0 means the conventional CE)
            normalize_length (bool): 
                True, normalize loss by sequence length; 
                False, normalize loss by batch size.
                Defaults to False.
        """
        super().__init__()
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.normalize_length = normalize_length
        self.criterion = nn.KLDivLoss(reduction="none")

    def forward(self, x: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        """Compute loss between x and target.
        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.
        
        Args:
            x (paddle.Tensor): prediction (batch, seqlen, class)
            target (paddle.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (paddle.Tensor) : The KL loss, scalar float value
        """
        B, T, D = paddle.shape(x)
        assert D == self.size
        x = x.reshape((-1, self.size))
        target = target.reshape([-1])

        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = paddle.full_like(x, self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)

        #target = target * (1 - ignore)  # avoid -1 index
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist += F.one_hot(target, self.size) * self.confidence

        kl = self.criterion(F.log_softmax(x, axis=1), true_dist)

        #TODO(Hui Zhang): sum not support bool type
        #total = len(target) - int(ignore.sum())
        total = len(target) - int(ignore.type_as(target).sum())
        denom = total if self.normalize_length else B
        #numer = (kl * (1 - ignore)).sum()
        numer = kl.masked_fill(ignore.unsqueeze(1), 0).sum()
        return numer / denom
