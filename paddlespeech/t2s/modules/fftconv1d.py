# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import math
import typing
from typing import Optional
from typing import Sequence

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...utils import satisfy_paddle_version

__all__ = [
    "fft_conv1d",
    "FFTConv1D",
]


def __unfold(x, kernel_size: int, stride: int):
    """1D only unfolding similar to the one from Paddlepaddle.

    Notes
    ------
    Given a tensor `x` of size `[*, T]` this will return
    a tensor `[*, F, K]` with `K` the kernel size, and `F` the number
    of frames. The i-th frame is a view onto `i * stride: i * stride + kernel_size`.
    This will automatically pad `x` to cover at least once all entries in `x`.

    Args:
        x (Tensor): 
            tensor for which to return the frames.
        kernel_size (int): 
            size of each frame.
        stride (int): 
            stride between each frame.
    """
    shape = list(x.shape)
    length = shape.pop(-1)
    n_frames = math.ceil((max(length, kernel_size) - kernel_size) / stride) + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    padded = F.pad(x, (0, tgt_length - length), data_format="NCL")
    strides: typing.List[int] = []
    for dim in range(padded.dim()):
        strides.append(padded.strides[dim])
    assert strides.pop(-1) == 1, "data should be contiguous"
    strides = strides + [stride, 1]
    return padded.as_strided(shape + [n_frames, kernel_size], strides)


def fft_conv1d(
        x: paddle.Tensor,
        weight: paddle.Tensor,
        bias: Optional[paddle.Tensor]=None,
        stride: int=1,
        padding: int=0,
        block_ratio: float=5, ):
    """
    Same as `paddle.nn.functional.conv1d` but using FFT for the convolution.
    Please check PaddlePaddle documentation for more information.

    Notes
    ------
    This function is faster than `paddle.nn.functional.conv1d` only in specific cases.
    Typically, the kernel size should be of the order of 256 to see any real gain,
    for a stride of 1.
    Dilation and groups are not supported at the moment. This function might use
    more memory than the default Conv1d implementation.

    Args:
        x (Tensor): 
            x signal of shape `[B, C, T]`.
        weight (Tensor): 
            weight of the convolution `[D, C, K]` with `D` the number of output channels.
        bias (Tensor or None): 
            if not None, bias term for the convolution.
        stride (int): 
            stride of convolution.
        padding (int): 
            padding to apply to x.
        block_ratio (float): 
            can be tuned for speed. x is splitted in chunks with a size of `int(block_ratio * kernel_size)`.

    Shape:

        - Inputs: `x` is `[B, C, T]`, `weight` is `[D, C, K]` and bias is `[D]`.
        - Output: `(*, T)`
    """
    x = F.pad(x, (padding, padding), data_format="NCL")
    batch, _, length = x.shape
    out_channels, _, kernel_size = weight.shape

    if length < kernel_size:
        raise RuntimeError(
            f"Input should be at least as large as the kernel size {kernel_size}, "
            f"but it is only {length} samples long.")
    if block_ratio < 1:
        raise RuntimeError("Block ratio must be greater than 1.")

    block_size: int = min(int(kernel_size * block_ratio), length)
    fold_stride = block_size - kernel_size + 1
    # weight = pad_to(weight, block_size)

    weight = F.pad(
        weight, (0, block_size - weight.shape[-1]),
        mode="constant",
        value=0.0,
        data_format="NCL")

    weight_z = paddle.fft.rfft(weight, axis=-1)

    # We pad `x` and get the different frames, on which
    frames = __unfold(x, block_size, fold_stride)

    frames_z = paddle.fft.rfft(frames, axis=-1)
    weight_z_coml = paddle.conj(weight_z)
    out_z = paddle.einsum("bcft,dct->bdft", frames_z, weight_z_coml)
    out = paddle.fft.irfft(out_z, n=block_size, axis=-1)

    # The last bit is invalid, because FFT will do a circular convolution.
    out = out[..., :-kernel_size + 1]
    out = out.reshape([batch, out_channels, -1])
    out = out[..., ::stride]
    target_length = (length - kernel_size) // stride + 1
    out = out[..., :target_length]
    if bias is not None:
        out += bias[:, None]
    return out


class FFTConv1D(paddle.nn.Layer):
    """
    Same as `paddle.nn.Conv1D` but based on a custom FFT-based convolution.
    Please check PaddlePaddle documentation for more information on `paddle.nn.Conv1D`.

    Notes
    ------
    This module is faster than `paddle.nn.Conv1D` only in specific cases.
    Typically, `kernel_size` should be of the order of 256 to see any real gain,
    for a stride of 1.
    Dilation and groups are not supported at the moment. This module might use
    more memory than the default Conv1D implementation.

    Args:
        in_channels (int): 
            number of `x` channels.
        out_channels (int): 
            number of output channels.
        kernel_size (int): 
            kernel size of convolution.
        stride (int): 
            stride of convolution.
        padding (int): 
            padding to apply to `x`.
        bias_attr (bool): 
            if True, use a bias term.

    Examples: 
        >>> fftconv = FFTConv1D(12, 24, 128, 4)
        >>> x = paddle.randn([4, 12, 1024])
        >>> print(list(fftconv(x).shape))
        [4, 24, 225]
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int=1,
            padding: int=0,
            bias_attr: bool=True, ):
        super(FFTConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Create a Conv1D layer to initialize weights and bias
        conv = paddle.nn.Conv1D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=bias_attr)
        self.weight = conv.weight
        if bias_attr:
            self.bias = conv.bias
        else:
            self.bias = None

    def forward(self, x: paddle.Tensor):
        return fft_conv1d(x, self.weight, self.bias, self.stride, self.padding)


# Currently, the API unfold in Paddle is extremely slow, so __unfold is implemented 
# using the `.strides` and `.as_strided` APIs. However, these are only supported in 
# Paddle version 2.6 and above, so F.conv1d and Conv1D are used as replacements.
if not satisfy_paddle_version('2.6'):
    fft_conv1d = F.conv1d
    FFTConv1D = nn.Conv1D
