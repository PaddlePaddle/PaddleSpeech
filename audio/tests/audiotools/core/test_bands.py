# MIT License, Copyright (c) 2020 Alexandre Défossez.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from julius(https://github.com/adefossez/julius/blob/main/tests/test_bands.py)
import random
import sys
import unittest

import paddle

from audio.audiotools.core import pure_tone
from audio.audiotools.core import split_bands
from audio.audiotools.core import SplitBands


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
