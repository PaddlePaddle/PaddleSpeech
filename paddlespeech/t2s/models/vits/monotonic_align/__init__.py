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
"""Maximum path calculation module.

This code is based on https://github.com/jaywalnut310/vits.

"""
import warnings

import numpy as np
import paddle
from numba import njit
from numba import prange

try:
    from .core import maximum_path_c

    is_cython_avalable = True
except ImportError:
    is_cython_avalable = False
    warnings.warn(
        "Cython version is not available. Fallback to 'EXPERIMETAL' numba version. "
        "If you want to use the cython version, please build it as follows: "
        "`cd paddlespeech/t2s/models/vits/monotonic_align; python setup.py build_ext --inplace`"
    )


def maximum_path(neg_x_ent: paddle.Tensor,
                 attn_mask: paddle.Tensor) -> paddle.Tensor:
    """Calculate maximum path.

    Args:
        neg_x_ent (Tensor): Negative X entropy tensor (B, T_feats, T_text).
        attn_mask (Tensor): Attention mask (B, T_feats, T_text).

    Returns:
        Tensor: Maximum path tensor (B, T_feats, T_text).

    """
    dtype = neg_x_ent.dtype
    neg_x_ent = neg_x_ent.numpy().astype(np.float32)
    path = np.zeros(neg_x_ent.shape, dtype=np.int32)
    t_t_max = attn_mask.sum(1)[:, 0].cpu().numpy().astype(np.int32)
    t_s_max = attn_mask.sum(2)[:, 0].cpu().numpy().astype(np.int32)
    if is_cython_avalable:
        maximum_path_c(path, neg_x_ent, t_t_max, t_s_max)
    else:
        maximum_path_numba(path, neg_x_ent, t_t_max, t_s_max)

    return paddle.cast(paddle.to_tensor(path), dtype=dtype)


@njit
def maximum_path_each_numba(path, value, t_y, t_x, max_neg_val=-np.inf):
    """Calculate a single maximum path with numba."""
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y - 1, x]
            if x == 0:
                if y == 0:
                    v_prev = 0.0
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or
                           value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1


@njit(parallel=True)
def maximum_path_numba(paths, values, t_ys, t_xs):
    """Calculate batch maximum path with numba."""
    for i in prange(paths.shape[0]):
        maximum_path_each_numba(paths[i], values[i], t_ys[i], t_xs[i])
