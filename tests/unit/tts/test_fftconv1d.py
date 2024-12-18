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
import sys
import unittest

import numpy as np
import paddle
from paddle.nn import Conv1D

from paddlespeech.t2s.modules import fft_conv1d
from paddlespeech.t2s.modules import FFTConv1D


class TestFFTConv1D(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.in_channels = 3
        self.out_channels = 16
        self.kernel_size = 5
        self.stride = 1
        self.padding = 1
        self.input_length = 32

    def _init_models(self, in_channels, out_channels, kernel_size, stride,
                     padding):
        x = paddle.randn([self.batch_size, in_channels, self.input_length])
        conv1d = paddle.nn.Conv1D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding)
        fft_conv1d = FFTConv1D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding)
        fft_conv1d.weight.set_value(conv1d.weight.numpy())
        if conv1d.bias is not None:
            fft_conv1d.bias.set_value(conv1d.bias.numpy())
        return x, conv1d, fft_conv1d

    def test_fft_conv1d_vs_conv1d_default(self):
        x, conv1d, fft_conv1d = self._init_models(
            self.in_channels, self.out_channels, self.kernel_size, self.stride,
            self.padding)
        out_conv1d = conv1d(x)
        out_fft_conv1d = fft_conv1d(x)
        self.assertTrue(
            np.allclose(out_conv1d.numpy(), out_fft_conv1d.numpy(), atol=1e-6))

    def test_fft_conv1d_vs_conv1d_no_padding(self):
        x, conv1d, fft_conv1d = self._init_models(
            self.in_channels, self.out_channels, self.kernel_size, self.stride,
            0)
        out_conv1d = conv1d(x)
        out_fft_conv1d = fft_conv1d(x)
        self.assertTrue(
            np.allclose(out_conv1d.numpy(), out_fft_conv1d.numpy(), atol=1e-6))

    def test_fft_conv1d_vs_conv1d_large_kernel(self):
        kernel_size = 256
        padding = kernel_size - 1
        x, conv1d, fft_conv1d = self._init_models(
            self.in_channels, self.out_channels, kernel_size, self.stride,
            padding)
        out_conv1d = conv1d(x)
        out_fft_conv1d = fft_conv1d(x)
        self.assertTrue(
            np.allclose(out_conv1d.numpy(), out_fft_conv1d.numpy(), atol=1e-6))

    def test_fft_conv1d_vs_conv1d_stride_2(self):
        x, conv1d, fft_conv1d = self._init_models(
            self.in_channels, self.out_channels, self.kernel_size, 2,
            self.padding)
        out_conv1d = conv1d(x)
        out_fft_conv1d = fft_conv1d(x)
        self.assertTrue(
            np.allclose(out_conv1d.numpy(), out_fft_conv1d.numpy(), atol=1e-6))

    def test_fft_conv1d_vs_conv1d_different_input_length(self):
        input_length = 1024
        x, conv1d, fft_conv1d = self._init_models(
            self.in_channels, self.out_channels, self.kernel_size, self.stride,
            self.padding)
        x = paddle.randn([self.batch_size, self.in_channels, input_length])
        out_conv1d = conv1d(x)
        out_fft_conv1d = fft_conv1d(x)
        self.assertTrue(
            np.allclose(out_conv1d.numpy(), out_fft_conv1d.numpy(), atol=1e-6))

    def test_fft_conv1d_vs_conv1d_no_bias(self):
        conv1d = paddle.nn.Conv1D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias_attr=False)
        fft_conv1d = FFTConv1D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias_attr=False)
        fft_conv1d.weight.set_value(conv1d.weight.numpy())
        x = paddle.randn([self.batch_size, self.in_channels, self.input_length])
        out_conv1d = conv1d(x)
        out_fft_conv1d = fft_conv1d(x)
        self.assertTrue(
            np.allclose(out_conv1d.numpy(), out_fft_conv1d.numpy(), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
