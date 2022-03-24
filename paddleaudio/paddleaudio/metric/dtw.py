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
import numpy as np
from dtaidistance import dtw_ndim

__all__ = [
    'dtw_distance',
]


def dtw_distance(xs: np.ndarray, ys: np.ndarray) -> float:
    """Dynamic Time Warping.
    This function keeps a compact matrix, not the full warping paths matrix.
    Uses dynamic programming to compute:

    Examples:
        .. code-block:: python

            wps[i, j] = (s1[i]-s2[j])**2 + min(
                            wps[i-1, j  ] + penalty,  // vertical   / insertion / expansion
                            wps[i  , j-1] + penalty,  // horizontal / deletion  / compression
                            wps[i-1, j-1])            // diagonal   / match

            dtw = sqrt(wps[-1, -1])

    Args:
        xs (np.ndarray): ref sequence, [T,D]
        ys (np.ndarray): hyp sequence, [T,D]

    Returns:
        float: dtw distance
    """
    return dtw_ndim.distance(xs, ys)
