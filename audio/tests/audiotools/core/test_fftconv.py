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
#
# This file contains code derived from [julius]
# (https://github.com/adefossez/julius),
# which is licensed under the MIT License:
#
# MIT License
#
# Copyright 2020 Alexandre Défossez
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import random
import sys
import unittest

import paddle
import paddle.nn.functional as F
sys.path.append("../..")
from audiotools.core import fft_conv1d, FFTConv1D

TOLERANCE = 1e-4  # as relative delta in percentage


class _BaseTest(unittest.TestCase):
    def setUp(self):
        paddle.seed(1234)
        random.seed(1234)

    def assertSimilar(self, a, b, msg=None, tol=TOLERANCE):
        delta = 100 * paddle.norm(a - b, p=2) / paddle.norm(b, p=2)
        self.assertLessEqual(delta.numpy(), tol, msg)

    def compare_paddle(self, *args, msg=None, tol=TOLERANCE, **kwargs):
        y_ref = F.conv1d(*args, **kwargs)
        y = fft_conv1d(*args, **kwargs)
        self.assertEqual(list(y.shape), list(y_ref.shape), msg)
        self.assertSimilar(y, y_ref, msg, tol)


class TestFFTConv1d(_BaseTest):
    def test_same_as_paddle(self):
        for _ in range(5):
            kernel_size = random.randrange(4, 128)
            batch_size = random.randrange(1, 6)
            length = random.randrange(kernel_size, 1024)
            chin = random.randrange(1, 12)
            chout = random.randrange(1, 12)
            bias = random.random() < 0.5
            if random.random() < 0.5:
                padding = 0
            else:
                padding = random.randrange(kernel_size // 2, 2 * kernel_size)
            x = paddle.randn([batch_size, chin, length])
            w = paddle.randn([chout, chin, kernel_size])
            keys = ["length", "kernel_size", "chin", "chout", "bias"]
            loc = locals()
            state = {key: loc[key] for key in keys}
            if bias:
                bias = paddle.randn([chout])
            else:
                bias = None
            for stride in [1, 2, 5]:
                state["stride"] = stride
                self.compare_paddle(
                    x, w, bias, stride, padding, msg=repr(state))

    def test_small_input(self):
        x = paddle.randn([1, 5, 19])
        w = paddle.randn([10, 5, 32])
        with self.assertRaises(RuntimeError):
            fft_conv1d(x, w)

        x = paddle.randn([1, 5, 19])
        w = paddle.randn([10, 5, 19])
        self.assertEqual(list(fft_conv1d(x, w).shape), [1, 10, 1])

    def test_module(self):
        x = paddle.randn([16, 4, 1024])
        mod = FFTConv1D(4, 5, 8, bias_attr=True)
        mod(x)
        mod = FFTConv1D(4, 5, 8, bias_attr=False)
        mod(x)

    def test_dynamic_graph(self):
        x = paddle.randn([16, 4, 1024])
        mod = FFTConv1D(4, 5, 8, bias_attr=True)
        self.assertEqual(list(mod(x).shape), [16, 5, 1024 - 8 + 1])


if __name__ == "__main__":
    unittest.main()
