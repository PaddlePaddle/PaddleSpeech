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

from deepspeech.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ["forced_align", "remove_duplicates_and_blank", "insert_blank"]


def remove_duplicates_and_blank(hyp: List[int], blank_id=0) -> List[int]:
    """ctc alignment to ctc label ids.

    "abaa-acee-" -> "abaace"

    Args:
        hyp (List[int]): hypotheses ids, (L)
        blank_id (int, optional): blank id. Defaults to 0.

    Returns:
        List[int]: remove dupicate ids, then remove blank id.
    """
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        # add non-blank into new_hyp
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        # skip repeat label
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def insert_blank(label: np.ndarray, blank_id: int=0) -> np.ndarray:
    """Insert blank token between every two label token.

    "abcdefg" -> "-a-b-c-d-e-f-g-"

    Args:
        label ([np.ndarray]): label ids, List[int], (L).
        blank_id (int, optional): blank id. Defaults to 0.

    Returns:
        [np.ndarray]: (2L+1).
    """
    label = np.expand_dims(label, 1)  #[L, 1]
    blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
    label = np.concatenate([blanks, label], axis=1)  #[L, 2]
    label = label.reshape(-1)  #[2L], -l-l-l
    label = np.append(label, label[0])  #[2L + 1], -l-l-l-
    return label


def forced_align(ctc_probs: paddle.Tensor, y: paddle.Tensor,
                 blank_id=0) -> List[int]:
    """ctc forced alignment.

    https://distill.pub/2017/ctc/

    Args:
        ctc_probs (paddle.Tensor): hidden state sequence, 2d tensor (T, D)
        y (paddle.Tensor): label id sequence tensor, 1d tensor (L)
        blank_id (int): blank symbol index
    Returns:
        List[int]: best alignment result, (T).
    """
    y_insert_blank = insert_blank(y, blank_id)  #(2L+1)

    log_alpha = paddle.zeros(
        (ctc_probs.size(0), len(y_insert_blank)))  #(T, 2L+1)
    log_alpha = log_alpha - float('inf')  # log of zero
    # TODO(Hui Zhang): zeros not support paddle.int16
    state_path = (paddle.zeros(
        (ctc_probs.size(0), len(y_insert_blank)), dtype=paddle.int32) - 1
                  )  # state path, Tuple((T, 2L+1))

    # init start state
    # TODO(Hui Zhang): VarBase.__getitem__() not support np.int64
    log_alpha[0, 0] = ctc_probs[0][int(y_insert_blank[0])]  # State-b, Sb
    log_alpha[0, 1] = ctc_probs[0][int(y_insert_blank[1])]  # State-nb, Snb

    for t in range(1, ctc_probs.size(0)):  # T
        for s in range(len(y_insert_blank)):  # 2L+1
            if y_insert_blank[s] == blank_id or s < 2 or y_insert_blank[
                    s] == y_insert_blank[s - 2]:
                candidates = paddle.to_tensor(
                    [log_alpha[t - 1, s], log_alpha[t - 1, s - 1]])
                prev_state = [s, s - 1]
            else:
                candidates = paddle.to_tensor([
                    log_alpha[t - 1, s],
                    log_alpha[t - 1, s - 1],
                    log_alpha[t - 1, s - 2],
                ])
                prev_state = [s, s - 1, s - 2]
            # TODO(Hui Zhang): VarBase.__getitem__() not support np.int64
            log_alpha[t, s] = paddle.max(candidates) + ctc_probs[t][int(
                y_insert_blank[s])]
            state_path[t, s] = prev_state[paddle.argmax(candidates)]

    # TODO(Hui Zhang): zeros not support paddle.int16
    state_seq = -1 * paddle.ones((ctc_probs.size(0), 1), dtype=paddle.int32)

    candidates = paddle.to_tensor([
        log_alpha[-1, len(y_insert_blank) - 1],  # Sb
        log_alpha[-1, len(y_insert_blank) - 2]  # Snb
    ])
    prev_state = [len(y_insert_blank) - 1, len(y_insert_blank) - 2]
    state_seq[-1] = prev_state[paddle.argmax(candidates)]
    for t in range(ctc_probs.size(0) - 2, -1, -1):
        state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

    output_alignment = []
    for t in range(0, ctc_probs.size(0)):
        output_alignment.append(y_insert_blank[state_seq[t, 0]])

    return output_alignment
