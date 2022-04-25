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
import time

import paddle


def collate_features(batch):
    # (key, feat, label)
    collate_start = time.time()
    keys = []
    feats = []
    labels = []
    lengths = []
    for sample in batch:
        keys.append(sample[0])
        feats.append(sample[1])
        labels.append(sample[2])
        lengths.append(sample[1].shape[0])

    max_length = max(lengths)
    for i in range(len(feats)):
        feats[i] = paddle.nn.functional.pad(
            feats[i], [0, max_length - feats[i].shape[0], 0, 0],
            data_format='NLC')

    return keys, paddle.stack(feats), paddle.to_tensor(
        labels), paddle.to_tensor(lengths)
