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
import numpy as np
import paddle
from paddle import Tensor


def expand(encodings: Tensor, durations: Tensor) -> Tensor:
    """
    encodings: (B, T, C)
    durations: (B, T)
    """
    batch_size, t_enc = durations.shape
    durations = durations.numpy()
    slens = np.sum(durations, -1)
    t_dec = np.max(slens)
    M = np.zeros([batch_size, t_dec, t_enc])
    for i in range(batch_size):
        k = 0
        for j in range(t_enc):
            d = durations[i, j]
            M[i, k:k + d, j] = 1
            k += d
    M = paddle.to_tensor(M, dtype=encodings.dtype)
    encodings = paddle.matmul(M, encodings)
    return encodings
