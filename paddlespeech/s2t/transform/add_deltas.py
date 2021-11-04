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
# Modified from espnet(https://github.com/espnet/espnet)
import numpy as np


def delta(feat, window):
    assert window > 0
    delta_feat = np.zeros_like(feat)
    for i in range(1, window + 1):
        delta_feat[:-i] += i * feat[i:]
        delta_feat[i:] += -i * feat[:-i]
        delta_feat[-i:] += i * feat[-1]
        delta_feat[:i] += -i * feat[0]
    delta_feat /= 2 * sum(i**2 for i in range(1, window + 1))
    return delta_feat


def add_deltas(x, window=2, order=2):
    """
    Args:
        x (np.ndarray): speech feat, (T, D).

    Return:
        np.ndarray: (T, (1+order)*D)
    """
    feats = [x]
    for _ in range(order):
        feats.append(delta(feats[-1], window))
    return np.concatenate(feats, axis=1)


class AddDeltas():
    def __init__(self, window=2, order=2):
        self.window = window
        self.order = order

    def __repr__(self):
        return "{name}(window={window}, order={order}".format(
            name=self.__class__.__name__, window=self.window, order=self.order)

    def __call__(self, x):
        return add_deltas(x, window=self.window, order=self.order)
