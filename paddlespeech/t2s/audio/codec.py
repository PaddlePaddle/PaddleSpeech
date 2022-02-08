# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import math

import numpy as np
import paddle


# x: [0: 2**bit-1], return: [-1, 1]
def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.


#x: [-1, 1], return: [0, 2**bits-1]
def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)


# y: [-1, 1], mu: 2**bits, return: [0, 2**bits-1]
# see https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
# be careful the input `mu` here, which is +1 than that of the link above
def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


# from_labels = True:
# y: [0: 2**bit-1], mu: 2**bits, return: [-1,1]
# from_labels = False:
# y: [-1, 1], return: [-1, 1]
def decode_mu_law(y, mu, from_labels=True):
    # TODO: get rid of log2 - makes no sense
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = paddle.sign(y) / mu * ((1 + mu)**paddle.abs(y) - 1)
    return x
