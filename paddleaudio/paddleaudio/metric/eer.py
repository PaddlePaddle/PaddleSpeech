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
from typing import List

import numpy as np
import paddle
from sklearn.metrics import roc_curve


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> List[float]:
    """Compute EER and return score threshold.

    Args:
        labels (np.ndarray): the trial label, shape: [N], one-dimention, N refer to the samples num
        scores (np.ndarray): the trial scores, shape: [N], one-dimention, N refer to the samples num

    Returns:
        List[float]: eer and the specific threshold
    """
    fpr, tpr, threshold = roc_curve(y_true=labels, y_score=scores)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold


def compute_minDCF(positive_scores,
                   negative_scores,
                   c_miss=1.0,
                   c_fa=1.0,
                   p_target=0.01):
    """
    This is modified from SpeechBrain
    https://github.com/speechbrain/speechbrain/blob/085be635c07f16d42cd1295045bc46c407f1e15b/speechbrain/utils/metric_stats.py#L509
    Computes the minDCF metric normally used to evaluate speaker verification
    systems. The min_DCF is the minimum of the following C_det function computed
    within the defined threshold range:

    C_det =  c_miss * p_miss * p_target + c_fa * p_fa * (1 -p_target)

    where p_miss is the missing probability and p_fa is the probability of having
    a false alarm.

    Args:
        positive_scores (Paddle.Tensor): The scores from entries of the same class.
        negative_scores (Paddle.Tensor): The scores from entries of different classes.
        c_miss (float, optional): Cost assigned to a missing error (default 1.0).
        c_fa (float, optional): Cost assigned to a false alarm (default 1.0).
        p_target (float, optional): Prior probability of having a target (default 0.01).

    Returns:
        List[float]: min dcf and the specific threshold
    """
    # Computing candidate thresholds
    if len(positive_scores.shape) > 1:
        positive_scores = positive_scores.squeeze()

    if len(negative_scores.shape) > 1:
        negative_scores = negative_scores.squeeze()

    thresholds = paddle.sort(paddle.concat([positive_scores, negative_scores]))
    thresholds = paddle.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds = paddle.sort(paddle.concat([thresholds, interm_thresholds]))

    # Computing False Rejection Rate (miss detection)
    positive_scores = paddle.concat(
        len(thresholds) * [positive_scores.unsqueeze(0)])
    pos_scores_threshold = positive_scores.transpose(perm=[1, 0]) <= thresholds
    p_miss = (pos_scores_threshold.sum(0)
              ).astype("float32") / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    # Computing False Acceptance Rate (false alarm)
    negative_scores = paddle.concat(
        len(thresholds) * [negative_scores.unsqueeze(0)])
    neg_scores_threshold = negative_scores.transpose(perm=[1, 0]) > thresholds
    p_fa = (neg_scores_threshold.sum(0)
            ).astype("float32") / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    c_min = paddle.min(c_det, axis=0)
    min_index = paddle.argmin(c_det, axis=0)
    return float(c_min), float(thresholds[min_index])
