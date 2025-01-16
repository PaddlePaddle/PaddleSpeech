# MIT License, Copyright (c) 2020 Alexandre Défossez.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from julius(https://github.com/adefossez/julius/blob/main/tests/test_filters.py)
import math
import random
import sys
import unittest

import paddle

from paddlespeech.audiotools.core import highpass_filter
from paddlespeech.audiotools.core import highpass_filters


def pure_tone(freq: float, sr: float=128, dur: float=4, device=None):
    """
    Return a pure tone, i.e. cosine.

    Args:
        freq (float): frequency (in Hz)
        sr (float): sample rate (in Hz)
        dur (float): duration (in seconds)
    """
    time = paddle.arange(int(sr * dur), dtype="float32") / sr
    return paddle.cos(2 * math.pi * freq * time)


def delta(a, b, ref, fraction=0.9):
    length = a.shape[-1]
    compare_length = int(length * fraction)
    offset = (length - compare_length) // 2
    a = a[..., offset:offset + length]
    b = b[..., offset:offset + length]
    # 计算绝对差值，均值，然后除以ref的标准差，乘以100
    return 100 * paddle.mean(paddle.abs(a - b)) / paddle.std(ref)


TOLERANCE = 1  # Tolerance to errors as percentage of the std of the input signal


class _BaseTest(unittest.TestCase):
    def assertSimilar(self, a, b, ref, msg=None, tol=TOLERANCE):
        self.assertLessEqual(delta(a, b, ref), tol, msg)


class TestHighPassFilters(_BaseTest):
    def setUp(self):
        paddle.seed(1234)
        random.seed(1234)

    def test_keep_or_kill(self):
        for _ in range(10):
            freq = random.uniform(0.01, 0.4)
            sr = 1024
            tone = pure_tone(freq * sr, sr=sr, dur=10)

            # For this test we accept 5% tolerance in amplitude, or -26dB in power.
            tol = 5
            zeros = 16

            # If cutoff frequency is under freq, output should be input
            y_pass = highpass_filter(tone, 0.9 * freq, zeros=zeros)
            self.assertSimilar(
                y_pass, tone, tone, f"freq={freq}, pass", tol=tol)

            # If cutoff frequency is over freq, output should be zero
            y_killed = highpass_filter(tone, 1.1 * freq, zeros=zeros)
            self.assertSimilar(
                y_killed, 0 * tone, tone, f"freq={freq}, kill", tol=tol)

    def test_fft_nofft(self):
        for _ in range(10):
            x = paddle.randn([1024])
            freq = random.uniform(0.01, 0.5)
            y_fft = highpass_filter(x, freq, fft=True)
            y_ref = highpass_filter(x, freq, fft=False)
            self.assertSimilar(y_fft, y_ref, x, f"freq={freq}", tol=0.01)

    def test_constant(self):
        x = paddle.ones([2048])
        for zeros in [4, 10]:
            for freq in [0.01, 0.1]:
                y_high = highpass_filter(x, freq, zeros=zeros)
                self.assertLessEqual(y_high.abs().mean(), 1e-6, (zeros, freq))

    def test_stride(self):
        x = paddle.randn([1024])

        y = highpass_filters(x, [0.1, 0.2], stride=1)[:, ::3]
        y2 = highpass_filters(x, [0.1, 0.2], stride=3)

        self.assertEqual(y.shape, y2.shape)
        self.assertSimilar(y, y2, x)

        y = highpass_filters(x, [0.1, 0.2], stride=1, pad=False)[:, ::3]
        y2 = highpass_filters(x, [0.1, 0.2], stride=3, pad=False)

        self.assertEqual(y.shape, y2.shape)
        self.assertSimilar(y, y2, x)


if __name__ == "__main__":
    unittest.main()
