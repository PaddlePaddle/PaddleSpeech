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
from typing import Callable

import mcd.metrics_fast as mt
import numpy as np
from mcd import dtw

__all__ = [
    'mcd_distance',
]


def mcd_distance(xs: np.ndarray, ys: np.ndarray, cost_fn: Callable=mt.logSpecDbDist) -> float:
    """Mel cepstral distortion (MCD), dtw distance.

    Dynamic Time Warping.
    Uses dynamic programming to compute:

    Examples:
        .. code-block:: python

            wps[i, j] = cost_fn(xs[i], ys[j]) + min(
                            wps[i-1, j  ],  // vertical   / insertion / expansion
                            wps[i  , j-1],  // horizontal / deletion  / compression
                            wps[i-1, j-1])  // diagonal   / match

            dtw = sqrt(wps[-1, -1])

    Cost Function:
    Examples:
        .. code-block:: python

            logSpecDbConst = 10.0 / math.log(10.0) * math.sqrt(2.0)

            def logSpecDbDist(x, y):
                diff = x - y
                return logSpecDbConst * math.sqrt(np.inner(diff, diff))

    Args:
        xs (np.ndarray): ref sequence, [T,D]
        ys (np.ndarray): hyp sequence, [T,D]
        cost_fn (Callable, optional): Cost function. Defaults to mt.logSpecDbDist.

    Returns:
        float: dtw distance
    """

    min_cost, path = dtw.dtw(xs, ys, cost_fn)
    return min_cost
