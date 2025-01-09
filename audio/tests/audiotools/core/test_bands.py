# MIT License, Copyright (c) 2020 Alexandre Défossez.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from julius(https://github.com/adefossez/julius/blob/main/tests/test_bands.py)
import random
import sys
import unittest

import paddle
sys.path.append("../..")
from audiotools.core import pure_tone, SplitBands, split_bands


def delta(a, b, ref, fraction=0.9):
    length = a.shape[-1]
    compare_length = int(length * fraction)
    offset = (length - compare_length) // 2
    a = a[..., offset:offset + length]
    b = b[..., offset:offset + length]
    return 100 * paddle.abs(a - b).mean() / ref.std()


TOLERANCE = 0.5  # Tolerance to errors as percentage of the std of the input signal


class _BaseTest(unittest.TestCase):
    def assertSimilar(self, a, b, ref, msg=None, tol=TOLERANCE):
        self.assertLessEqual(delta(a, b, ref), tol, msg)


class TestLowPassFilters(_BaseTest):
    def setUp(self):
        paddle.seed(1234)
        random.seed(1234)

    def test_keep_or_kill(self):
        sr = 256
        low = pure_tone(10, sr)
        mid = pure_tone(40, sr)
        high = pure_tone(100, sr)

        x = low + mid + high

        decomp = split_bands(x, sr, cutoffs=[20, 70])
        self.assertEqual(len(decomp), 3)
        for est, gt, name in zip(decomp, [low, mid, high],
                                 ["low", "mid", "high"]):
            self.assertSimilar(est, gt, gt, name)


if __name__ == "__main__":
    unittest.main()
