# File under the MIT license, see https://github.com/your_repo/your_license for details.
# Author: your_name, current_year
import random
import sys
import unittest

import paddle
import paddle.nn.functional as F
sys.path.append("../..")
from audiotools.core import fft_conv1d, FFTConv1d

TOLERANCE = 1e-4  # as relative delta in percentage


class _BaseTest(unittest.TestCase):
    def setUp(self):
        paddle.seed(1234)
        random.seed(1234)

    def assertSimilar(self, a, b, msg=None, tol=TOLERANCE):
        delta = 100 * paddle.norm(a - b, p=2) / paddle.norm(b, p=2)
        self.assertLessEqual(delta.numpy(), tol, msg)

    def compare_paddle(self,
                       *args,
                       block_ratio=10,
                       msg=None,
                       tol=TOLERANCE,
                       **kwargs):
        y_ref = F.conv1d(*args, **kwargs)
        y = fft_conv1d(*args, block_ratio=block_ratio, **kwargs)
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
            block_ratio = random.choice([5, 10, 20])
            bias = random.random() < 0.5
            if random.random() < 0.5:
                padding = 0
            else:
                padding = random.randrange(kernel_size // 2, 2 * kernel_size)
            x = paddle.randn([batch_size, chin, length])
            w = paddle.randn([chout, chin, kernel_size])
            keys = [
                "length", "kernel_size", "chin", "chout", "block_ratio", "bias"
            ]
            loc = locals()
            state = {key: loc[key] for key in keys}
            if bias:
                bias = paddle.randn([chout])
            else:
                bias = None
            for stride in [1, 2, 5]:
                state["stride"] = stride
                self.compare_paddle(
                    x,
                    w,
                    bias,
                    stride,
                    padding,
                    block_ratio=block_ratio,
                    msg=repr(state))

    def test_small_input(self):
        x = paddle.randn([1, 5, 19])
        w = paddle.randn([10, 5, 32])
        with self.assertRaises(RuntimeError):
            fft_conv1d(x, w)

        x = paddle.randn([1, 5, 19])
        w = paddle.randn([10, 5, 19])
        self.assertEqual(list(fft_conv1d(x, w).shape), [1, 10, 1])

    def test_block_ratio(self):
        x = paddle.randn([1, 5, 1024])
        w = paddle.randn([10, 5, 19])
        ref = fft_conv1d(x, w)
        for block_ratio in [1, 5, 10, 20]:
            y = fft_conv1d(x, w, block_ratio=block_ratio)
            self.assertSimilar(y, ref, msg=str(block_ratio))

        with self.assertRaises(RuntimeError):
            y = fft_conv1d(x, w, block_ratio=0.9)

    def test_module(self):
        x = paddle.randn([16, 4, 1024])
        mod = FFTConv1d(4, 5, 8, bias=True)
        mod(x)
        mod = FFTConv1d(4, 5, 8, bias=False)
        mod(x)

    def test_dynamic_graph(self):
        x = paddle.randn([16, 4, 1024])
        mod = FFTConv1d(4, 5, 8, bias=True)
        self.assertEqual(list(mod(x).shape), [16, 5, 1024 - 8 + 1])


if __name__ == "__main__":
    unittest.main()
