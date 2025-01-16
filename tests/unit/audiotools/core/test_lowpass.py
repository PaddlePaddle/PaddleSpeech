# MIT License, Copyright (c) 2020 Alexandre Défossez.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from julius(https://github.com/adefossez/julius/blob/main/tests/test_lowpass.py)
import math
import random
import sys
import unittest

import numpy as np
import paddle

from paddlespeech.audiotools.core import lowpass_filter
from paddlespeech.audiotools.core import LowPassFilter
from paddlespeech.audiotools.core import LowPassFilters
from paddlespeech.audiotools.core import resample_frac


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


class TestLowPassFilters(_BaseTest):
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

            # If cutoff frequency is under freq, output should be zero
            y_killed = lowpass_filter(tone, 0.9 * freq, zeros=zeros)
            self.assertSimilar(
                y_killed, 0 * y_killed, tone, f"freq={freq}, kill", tol=tol)

            # If cutoff frequency is under freq, output should be input
            y_pass = lowpass_filter(tone, 1.1 * freq, zeros=zeros)
            self.assertSimilar(
                y_pass, tone, tone, f"freq={freq}, pass", tol=tol)

    def test_same_as_downsample(self):
        for _ in range(10):
            x = paddle.randn([2 * 3 * 4 * 100])
            x = paddle.ones_like(x)
            np.random.seed(1234)
            x = paddle.to_tensor(
                np.random.randn(2 * 3 * 4 * 100), dtype="float32")
            rolloff = 0.945
            for old_sr in [2, 3, 4]:
                y_resampled = resample_frac(
                    x, old_sr, 1, rolloff=rolloff, zeros=16)
                y_lowpass = lowpass_filter(
                    x, rolloff / old_sr / 2, stride=old_sr, zeros=16)
                self.assertSimilar(y_resampled, y_lowpass, x,
                                   f"old_sr={old_sr}")

    def test_fft_nofft(self):
        for _ in range(10):
            x = paddle.randn([1024])
            freq = random.uniform(0.01, 0.5)
            y_fft = lowpass_filter(x, freq, fft=True)
            y_ref = lowpass_filter(x, freq, fft=False)
            self.assertSimilar(y_fft, y_ref, x, f"freq={freq}", tol=0.01)

    def test_constant(self):
        x = paddle.ones([2048])
        for zeros in [4, 10]:
            for freq in [0.01, 0.1]:
                y_low = lowpass_filter(x, freq, zeros=zeros)
                self.assertLessEqual((y_low - 1).abs().mean(), 1e-6,
                                     (zeros, freq))


if __name__ == "__main__":
    unittest.main()
