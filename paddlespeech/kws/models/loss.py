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
# Modified from wekws(https://github.com/wenet-e2e/wekws)
import paddle


def fill_mask_elements(condition, value, x):
    assert condition.shape == x.shape
    values = paddle.ones_like(x, dtype=x.dtype) * value
    return paddle.where(condition, values, x)


def max_pooling_loss(logits: paddle.Tensor,
                     target: paddle.Tensor,
                     lengths: paddle.Tensor,
                     min_duration: int=0):

    mask = padding_mask(lengths)
    num_utts = logits.shape[0]
    num_keywords = logits.shape[2]

    loss = 0.0
    for i in range(num_utts):
        for j in range(num_keywords):
            # Add entropy loss CE = -(t * log(p) + (1 - t) * log(1 - p))
            if target[i] == j:
                # For the keyword, do max-polling
                prob = logits[i, :, j]
                m = mask[i]
                if min_duration > 0:
                    m[:min_duration] = True
                prob = fill_mask_elements(m, 0.0, prob)
                prob = paddle.clip(prob, 1e-8, 1.0)
                max_prob = prob.max()
                loss += -paddle.log(max_prob)
            else:
                # For other keywords or filler, do min-polling
                prob = 1 - logits[i, :, j]
                prob = fill_mask_elements(mask[i], 1.0, prob)
                prob = paddle.clip(prob, 1e-8, 1.0)
                min_prob = prob.min()
                loss += -paddle.log(min_prob)
    loss = loss / num_utts

    # Compute accuracy of current batch
    mask = mask.unsqueeze(-1)
    logits = fill_mask_elements(mask, 0.0, logits)
    max_logits = logits.max(1)
    num_correct = 0
    for i in range(num_utts):
        max_p = max_logits[i].max(0).item()
        idx = max_logits[i].argmax(0).item()
        # Predict correct as the i'th keyword
        if max_p > 0.5 and idx == target[i].item():
            num_correct += 1
        # Predict correct as the filler, filler id < 0
        if max_p < 0.5 and target[i].item() < 0:
            num_correct += 1
    acc = num_correct / num_utts
    # acc = 0.0
    return loss, num_correct, acc


def padding_mask(lengths: paddle.Tensor) -> paddle.Tensor:
    batch_size = lengths.shape[0]
    max_len = int(lengths.max().item())
    seq = paddle.arange(max_len, dtype=paddle.int64)
    seq = seq.expand((batch_size, max_len))
    return seq >= lengths.unsqueeze(1)
