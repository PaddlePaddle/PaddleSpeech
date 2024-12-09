# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020
"""
Implementation of a FFT based 1D convolution in PaddlePaddle.
While FFT is used in some cases for small kernel sizes, it is not the default for long ones, e.g. 512.
This module implements efficient FFT based convolutions for such cases. A typical
application is for evaluating FIR filters with a long receptive field, typically
evaluated with a stride of 1.
"""
import math
import typing
from typing import Optional
from typing import Sequence

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .resample import sinc


def pad_to(tensor: paddle.Tensor,
           target_length: int,
           mode: str="constant",
           value: float=0.0):
    """
    Pad the given tensor to the given length, with 0s on the right.
    """
    return F.pad(
        tensor, (0, target_length - tensor.shape[-1]),
        mode=mode,
        value=value,
        data_format="NCL")


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


def unfold(_input, kernel_size: int, stride: int):
    """1D only unfolding similar to the one from PyTorch.
    However PyTorch unfold is extremely slow.

    Given an _input tensor of size `[*, T]` this will return
    a tensor `[*, F, K]` with `K` the kernel size, and `F` the number
    of frames. The i-th frame is a view onto `i * stride: i * stride + kernel_size`.
    This will automatically pad the _input to cover at least once all entries in `_input`.

    Args:
        _input (Tensor): tensor for which to return the frames.
        kernel_size (int): size of each frame.
        stride (int): stride between each frame.

    Shape:

        - Inputs: `_input` is `[*, T]`
        - Output: `[*, F, kernel_size]` with `F = 1 + ceil((T - kernel_size) / stride)`


    ..Warning:: unlike PyTorch unfold, this will pad the _input
        so that any position in `_input` is covered by at least one frame.
    """
    shape = list(_input.shape)
    length = shape.pop(-1)
    n_frames = math.ceil((max(length, kernel_size) - kernel_size) / stride) + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    padded = F.pad(_input, (0, tgt_length - length), data_format="NCL")
    strides: typing.List[int] = []
    for dim in range(padded.dim()):
        strides.append(padded.strides[dim])
    assert strides.pop(-1) == 1, "data should be contiguous"
    strides = strides + [stride, 1]
    return padded.as_strided(shape + [n_frames, kernel_size], strides)


def _new_rfft(x: paddle.Tensor):
    z = paddle.fft.rfft(x, axis=-1)

    z_real = paddle.real(z)
    z_imag = paddle.imag(z)

    z_view_as_real = paddle.stack([z_real, z_imag], axis=-1)
    return z_view_as_real


def _new_irfft(x: paddle.Tensor, length: int):
    x_real = x[..., 0]
    x_imag = x[..., 1]
    x_view_as_complex = paddle.complex(x_real, x_imag)
    return paddle.fft.irfft(x_view_as_complex, n=length, axis=-1)


def _compl_mul_conjugate(a: paddle.Tensor, b: paddle.Tensor):
    """
    Given a and b two tensors of dimension 4
    with the last dimension being the real and imaginary part,
    returns a multiplied by the conjugate of b, the multiplication
    being with respect to the second dimension.

    PaddlePaddle does not have direct support for complex number operations
    using einsum in the same manner as PyTorch, but we can manually compute
    the equivalent result.
    """
    # Extract the real and imaginary parts of a and b
    real_a = a[..., 0]
    imag_a = a[..., 1]
    real_b = b[..., 0]
    imag_b = b[..., 1]

    # Compute the multiplication with respect to the second dimension manually
    real_part = paddle.einsum("bcft,dct->bdft", real_a, real_b) + paddle.einsum(
        "bcft,dct->bdft", imag_a, imag_b)
    imag_part = paddle.einsum("bcft,dct->bdft", imag_a, real_b) - paddle.einsum(
        "bcft,dct->bdft", real_a, imag_b)

    # Stack the real and imaginary parts together
    result = paddle.stack([real_part, imag_part], axis=-1)
    return result


def fft_conv1d(
        _input: paddle.Tensor,
        weight: paddle.Tensor,
        bias: Optional[paddle.Tensor]=None,
        stride: int=1,
        padding: int=0,
        block_ratio: float=5, ):
    """
    Same as `paddle.nn.functional.conv1d` but using FFT for the convolution.
    Please check PaddlePaddle documentation for more information.

    Args:
        _input (Tensor): _input signal of shape `[B, C, T]`.
        weight (Tensor): weight of the convolution `[D, C, K]` with `D` the number
            of output channels.
        bias (Tensor or None): if not None, bias term for the convolution.
        stride (int): stride of convolution.
        padding (int): padding to apply to the _input.
        block_ratio (float): can be tuned for speed. The _input is splitted in chunks
            with a size of `int(block_ratio * kernel_size)`.

    Shape:

        - Inputs: `_input` is `[B, C, T]`, `weight` is `[D, C, K]` and bias is `[D]`.
        - Output: `(*, T)`


    ..note::
        This function is faster than `paddle.nn.functional.conv1d` only in specific cases.
        Typically, the kernel size should be of the order of 256 to see any real gain,
        for a stride of 1.

    ..Warning::
        Dilation and groups are not supported at the moment. This function might use
        more memory than the default Conv1d implementation.
    """
    _input = F.pad(_input, (padding, padding), data_format="NCL")
    batch, channels, length = _input.shape
    out_channels, _, kernel_size = weight.shape

    if length < kernel_size:
        raise RuntimeError(
            f"Input should be at least as large as the kernel size {kernel_size}, "
            f"but it is only {length} samples long.")
    if block_ratio < 1:
        raise RuntimeError("Block ratio must be greater than 1.")

    block_size: int = min(int(kernel_size * block_ratio), length)
    fold_stride = block_size - kernel_size + 1
    weight = pad_to(weight, block_size)
    weight_z = _new_rfft(weight)

    # We pad the _input and get the different frames, on which
    frames = unfold(_input, block_size, fold_stride)

    frames_z = _new_rfft(frames)
    out_z = _compl_mul_conjugate(frames_z, weight_z)
    out = _new_irfft(out_z, block_size)
    # The last bit is invalid, because FFT will do a circular convolution.
    out = out[..., :-kernel_size + 1]
    out = out.reshape([batch, out_channels, -1])
    out = out[..., ::stride]
    target_length = (length - kernel_size) // stride + 1
    out = out[..., :target_length]
    if bias is not None:
        out += bias[:, None]
    return out


class FFTConv1d(paddle.nn.Layer):
    """
    Same as `paddle.nn.Conv1D` but based on a custom FFT-based convolution.
    Please check PaddlePaddle documentation for more information on `paddle.nn.Conv1D`.

    Args:
        in_channels (int): number of _input channels.
        out_channels (int): number of output channels.
        kernel_size (int): kernel size of convolution.
        stride (int): stride of convolution.
        padding (int): padding to apply to the _input.
        bias (bool): if True, use a bias term.

    ..note::
        This module is faster than `paddle.nn.Conv1D` only in specific cases.
        Typically, `kernel_size` should be of the order of 256 to see any real gain,
        for a stride of 1.

    ..warning::
        Dilation and groups are not supported at the moment. This module might use
        more memory than the default Conv1D implementation.

    >>> fftconv = FFTConv1d(12, 24, 128, 4)
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
            bias: bool=True, ):
        super(FFTConv1d, self).__init__()
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
            bias_attr=bias)
        self.weight = conv.weight
        if bias:
            self.bias = conv.bias
        else:
            self.bias = None

    def forward(self, _input: paddle.Tensor):
        return fft_conv1d(_input, self.weight, self.bias, self.stride,
                          self.padding)


class LowPassFilters(nn.Layer):
    """
    Bank of low pass filters.
    """

    def __init__(self,
                 cutoffs: Sequence[float],
                 stride: int=1,
                 pad: bool=True,
                 zeros: float=8,
                 fft: Optional[bool]=None,
                 dtype="float32"):
        super(LowPassFilters, self).__init__()
        self.cutoffs = list(cutoffs)
        if min(self.cutoffs) < 0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if max(self.cutoffs) > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.stride = stride
        self.pad = pad
        self.zeros = zeros
        self.half_size = int(zeros / min([c for c in self.cutoffs if c > 0]) /
                             2)
        if fft is None:
            fft = self.half_size > 32
        self.fft = fft

        # Create filters
        window = paddle.audio.functional.get_window(
            "hann", 2 * self.half_size + 1, fftbins=False, dtype=dtype)
        time = paddle.arange(
            -self.half_size, self.half_size + 1, dtype="float32")
        filters = []
        for cutoff in cutoffs:
            if cutoff == 0:
                filter_ = paddle.zeros_like(time)
            else:
                filter_ = 2 * cutoff * window * sinc(2 * cutoff * math.pi *
                                                     time)
                # Normalize filter
                filter_ /= paddle.sum(filter_)
            filters.append(filter_)
        filters = paddle.stack(filters)[:, None]
        self.filters = self.create_parameter(
            shape=filters.shape,
            default_initializer=nn.initializer.Constant(value=0.0),
            dtype="float32",
            is_bias=False,
            attr=paddle.ParamAttr(trainable=False), )
        self.filters.set_value(filters)

    def forward(self, _input):
        shape = list(_input.shape)
        _input = _input.reshape([-1, 1, shape[-1]])
        if self.pad:
            _input = F.pad(
                _input, (self.half_size, self.half_size),
                mode="replicate",
                data_format="NCL")
        if self.fft:
            out = fft_conv1d(_input, self.filters, stride=self.stride)
        else:
            out = F.conv1d(_input, self.filters, stride=self.stride)

        shape.insert(0, len(self.cutoffs))
        shape[-1] = out.shape[-1]
        return out.transpose([1, 0, 2]).reshape(shape)


class LowPassFilter(nn.Layer):
    """
    Same as `LowPassFilters` but applies a single low pass filter.
    """

    def __init__(self,
                 cutoff: float,
                 stride: int=1,
                 pad: bool=True,
                 zeros: float=8,
                 fft: Optional[bool]=None):
        super(LowPassFilter, self).__init__()
        self._lowpasses = LowPassFilters([cutoff], stride, pad, zeros, fft)

    @property
    def cutoff(self):
        return self._lowpasses.cutoffs[0]

    @property
    def stride(self):
        return self._lowpasses.stride

    @property
    def pad(self):
        return self._lowpasses.pad

    @property
    def zeros(self):
        return self._lowpasses.zeros

    @property
    def fft(self):
        return self._lowpasses.fft

    def forward(self, _input):
        return self._lowpasses(_input)[0]


def lowpass_filters(
        _input: paddle.Tensor,
        cutoffs: Sequence[float],
        stride: int=1,
        pad: bool=True,
        zeros: float=8,
        fft: Optional[bool]=None, ):
    """
    Functional version of `LowPassFilters`, refer to this class for more information.
    """
    return LowPassFilters(cutoffs, stride, pad, zeros, fft)(_input)


def lowpass_filter(_input: paddle.Tensor,
                   cutoff: float,
                   stride: int=1,
                   pad: bool=True,
                   zeros: float=8,
                   fft: Optional[bool]=None):
    """
    Same as `lowpass_filters` but with a single cutoff frequency.
    Output will not have a dimension inserted in the front.
    """
    return lowpass_filters(_input, [cutoff], stride, pad, zeros, fft)[0]


class HighPassFilters(paddle.nn.Layer):
    """
    Bank of high pass filters. See `julius.lowpass.LowPassFilters` for more
    details on the implementation.

    Args:
        cutoffs (list[float]): list of cutoff frequencies, in [0, 0.5] expressed as `f/f_s` where
            f_s is the samplerate and `f` is the cutoff frequency.
            The upper limit is 0.5, because a signal sampled at `f_s` contains only
            frequencies under `f_s / 2`.
        stride (int): how much to decimate the output. Probably not a good idea
            to do so with a high pass filters though...
        pad (bool): if True, appropriately pad the _input with zero over the edge. If `stride=1`,
            the output will have the same length as the _input.
        zeros (float): Number of zero crossings to keep.
            Controls the receptive field of the Finite Impulse Response filter.
            For filters with low cutoff frequency, e.g. 40Hz at 44.1kHz,
            it is a bad idea to set this to a high value.
            This is likely appropriate for most use. Lower values
            will result in a faster filter, but with a slower attenuation around the
            cutoff frequency.
        fft (bool or None): if True, uses `julius.fftconv` rather than PyTorch convolutions.
            If False, uses PyTorch convolutions. If None, either one will be chosen automatically
            depending on the effective filter size.


    ..warning::
        All the filters will use the same filter size, aligned on the lowest
        frequency provided. If you combine a lot of filters with very diverse frequencies, it might
        be more efficient to split them over multiple modules with similar frequencies.

    Shape:

        - Input: `[*, T]`
        - Output: `[F, *, T']`, with `T'=T` if `pad` is True and `stride` is 1, and
            `F` is the numer of cutoff frequencies.

    >>> highpass = HighPassFilters([1/4])
    >>> x = paddle.randn([4, 12, 21, 1024])
    >>> list(highpass(x).shape)
    [1, 4, 12, 21, 1024]
    """

    def __init__(self,
                 cutoffs: Sequence[float],
                 stride: int=1,
                 pad: bool=True,
                 zeros: float=8,
                 fft: Optional[bool]=None):
        super().__init__()
        self._lowpasses = LowPassFilters(cutoffs, stride, pad, zeros, fft)

    @property
    def cutoffs(self):
        return self._lowpasses.cutoffs

    @property
    def stride(self):
        return self._lowpasses.stride

    @property
    def pad(self):
        return self._lowpasses.pad

    @property
    def zeros(self):
        return self._lowpasses.zeros

    @property
    def fft(self):
        return self._lowpasses.fft

    def forward(self, _input):
        lows = self._lowpasses(_input)

        # We need to extract the right portion of the _input in case
        # pad is False or stride > 1
        if self.pad:
            start, end = 0, _input.shape[-1]
        else:
            start = self._lowpasses.half_size
            end = -start
        _input = _input[..., start:end:self.stride]
        highs = _input - lows
        return highs


class HighPassFilter(paddle.nn.Layer):
    """
    Same as `HighPassFilters` but applies a single high pass filter.

    Shape:

        - Input: `[*, T]`
        - Output: `[*, T']`, with `T'=T` if `pad` is True and `stride` is 1.

    >>> highpass = HighPassFilter(1/4, stride=1)
    >>> x = paddle.randn([4, 124])
    >>> list(highpass(x).shape)
    [4, 124]
    """

    def __init__(self,
                 cutoff: float,
                 stride: int=1,
                 pad: bool=True,
                 zeros: float=8,
                 fft: Optional[bool]=None):
        super().__init__()
        self._highpasses = HighPassFilters([cutoff], stride, pad, zeros, fft)

    @property
    def cutoff(self):
        return self._highpasses.cutoffs[0]

    @property
    def stride(self):
        return self._highpasses.stride

    @property
    def pad(self):
        return self._highpasses.pad

    @property
    def zeros(self):
        return self._highpasses.zeros

    @property
    def fft(self):
        return self._highpasses.fft

    def forward(self, _input):
        return self._highpasses(_input)[0]


def highpass_filters(
        _input: paddle.Tensor,
        cutoffs: Sequence[float],
        stride: int=1,
        pad: bool=True,
        zeros: float=8,
        fft: Optional[bool]=None, ):
    """
    Functional version of `HighPassFilters`, refer to this class for more information.
    """
    return HighPassFilters(cutoffs, stride, pad, zeros, fft)(_input)


def highpass_filter(_input: paddle.Tensor,
                    cutoff: float,
                    stride: int=1,
                    pad: bool=True,
                    zeros: float=8,
                    fft: Optional[bool]=None):
    """
    Functional version of `HighPassFilter`, refer to this class for more information.
    Output will not have a dimension inserted in the front.
    """
    return highpass_filters(_input, [cutoff], stride, pad, zeros, fft)[0]


class SplitBands(paddle.nn.Layer):
    """
    Decomposes a signal over the given frequency bands in the waveform domain using
    a cascade of low pass filters as implemented by `julius.lowpass.LowPassFilters`.
    You can either specify explicitly the frequency cutoffs, or just the number of bands,
    in which case the frequency cutoffs will be spread out evenly in mel scale.

    Args:
        sample_rate (float): Sample rate of the input signal in Hz.
        n_bands (int or None): number of bands, when not giving them explicitly with `cutoffs`.
            In that case, the cutoff frequencies will be evenly spaced in mel-space.
        cutoffs (list[float] or None): list of frequency cutoffs in Hz.
        pad (bool): if True, appropriately pad the input with zero over the edge. If `stride=1`,
            the output will have the same length as the input.
        zeros (float): Number of zero crossings to keep. See `LowPassFilters` for more informations.
        fft (bool or None): See `LowPassFilters` for more info.

    ..note::
        The sum of all the bands will always be the input signal.

    ..warning::
        Unlike `julius.lowpass.LowPassFilters`, the cutoffs frequencies must be provided in Hz along
        with the sample rate.

    Shape:

        - Input: `[*, T]`
        - Output: `[B, *, T']`, with `T'=T` if `pad` is True.
            If `n_bands` was provided, `B = n_bands` otherwise `B = len(cutoffs) + 1`

    >>> bands = SplitBands(sample_rate=128, n_bands=10)
    >>> x = paddle.randn(shape=[6, 4, 1024])
    >>> list(bands(x).shape)
    [10, 6, 4, 1024]
    """

    def __init__(
            self,
            sample_rate: float,
            n_bands: Optional[int]=None,
            cutoffs: Optional[Sequence[float]]=None,
            pad: bool=True,
            zeros: float=8,
            fft: Optional[bool]=None, ):
        super(SplitBands, self).__init__()
        if (cutoffs is None) + (n_bands is None) != 1:
            raise ValueError(
                "You must provide either n_bands, or cutoffs, but not both.")

        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self._cutoffs = list(cutoffs) if cutoffs is not None else None
        self.pad = pad
        self.zeros = zeros
        self.fft = fft

        if cutoffs is None:
            if n_bands is None:
                raise ValueError("You must provide one of n_bands or cutoffs.")
            if not n_bands >= 1:
                raise ValueError(
                    f"n_bands must be greater than one (got {n_bands})")
            cutoffs = paddle.audio.functional.mel_frequencies(
                n_bands + 1, 0, sample_rate / 2)[1:-1]
        else:
            if max(cutoffs) > 0.5 * sample_rate:
                raise ValueError(
                    "A cutoff above sample_rate/2 does not make sense.")
        if len(cutoffs) > 0:
            self.lowpass = LowPassFilters(
                [c / sample_rate for c in cutoffs],
                pad=pad,
                zeros=zeros,
                fft=fft)
        else:
            self.lowpass = None  # type: ignore

    def forward(self, input):
        if self.lowpass is None:
            return input[None]
        lows = self.lowpass(input)
        low = lows[0]
        bands = [low]
        for low_and_band in lows[1:]:
            # Get a bandpass filter by subtracting lowpasses
            band = low_and_band - low
            bands.append(band)
            low = low_and_band
        # Last band is whatever is left in the signal
        bands.append(input - low)
        return paddle.stack(bands)

    @property
    def cutoffs(self):
        if self._cutoffs is not None:
            return self._cutoffs
        elif self.lowpass is not None:
            return [c * self.sample_rate for c in self.lowpass.cutoffs]
        else:
            return []


def split_bands(
        signal: paddle.Tensor,
        sample_rate: float,
        n_bands: Optional[int]=None,
        cutoffs: Optional[Sequence[float]]=None,
        pad: bool=True,
        zeros: float=8,
        fft: Optional[bool]=None, ):
    """
    Functional version of `SplitBands`, refer to this class for more information.

    >>> x = paddle.randn(shape=[6, 4, 1024])
    >>> list(split_bands(x, sample_rate=64, cutoffs=[12, 24]).shape)
    [3, 6, 4, 1024]
    """
    return SplitBands(sample_rate, n_bands, cutoffs, pad, zeros, fft)(signal)
