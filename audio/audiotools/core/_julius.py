# MIT License, Copyright (c) 2020 Alexandre Défossez.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# 
# Modified from julius(https://github.com/adefossez/julius/tree/main/julius)
"""
Implementation of a FFT based 1D convolution in PaddlePaddle.
While FFT is used in some cases for small kernel sizes, it is not the default for long ones, e.g. 512.
This module implements efficient FFT based convolutions for such cases. A typical
application is for evaluating FIR filters with a long receptive field, typically
evaluated with a stride of 1.
"""
import inspect
import math
import sys
import typing
from typing import Optional
from typing import Sequence

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlespeech.t2s.modules import fft_conv1d
from paddlespeech.t2s.modules import FFTConv1D
from paddlespeech.utils import satisfy_paddle_version

__all__ = [
    'highpass_filter', 'highpass_filters', 'lowpass_filter', 'LowPassFilter',
    'LowPassFilters', 'pure_tone', 'resample_frac', 'split_bands', 'SplitBands'
]


def simple_repr(obj, attrs: Optional[Sequence[str]]=None, overrides: dict={}):
    """
    Return a simple representation string for `obj`.
    If `attrs` is not None, it should be a list of attributes to include.
    """
    params = inspect.signature(obj.__class__).parameters
    attrs_repr = []
    if attrs is None:
        attrs = list(params.keys())
    for attr in attrs:
        display = False
        if attr in overrides:
            value = overrides[attr]
        elif hasattr(obj, attr):
            value = getattr(obj, attr)
        else:
            continue
        if attr in params:
            param = params[attr]
            if param.default is inspect._empty or value != param.default:  # type: ignore
                display = True
        else:
            display = True

        if display:
            attrs_repr.append(f"{attr}={value}")
    return f"{obj.__class__.__name__}({','.join(attrs_repr)})"


def sinc(x: paddle.Tensor):
    """
    Implementation of sinc, i.e. sin(x) / x

    __Warning__: the input is not multiplied by `pi`!
    """
    if satisfy_paddle_version("3.0"):
        return paddle.sinc(x)

    return paddle.where(
        x == 0,
        paddle.to_tensor(1.0, dtype=x.dtype, place=x.place),
        paddle.sin(x) / x, )


class ResampleFrac(paddle.nn.Layer):
    """
    Resampling from the sample rate `old_sr` to `new_sr`.
    """

    def __init__(self,
                 old_sr: int,
                 new_sr: int,
                 zeros: int=24,
                 rolloff: float=0.945):
        """
        Args:
            old_sr (int): sample rate of the input signal x.
            new_sr (int): sample rate of the output.
            zeros (int): number of zero crossing to keep in the sinc filter.
            rolloff (float): use a lowpass filter that is `rolloff * new_sr / 2`,
                to ensure sufficient margin due to the imperfection of the FIR filter used.
                Lowering this value will reduce anti-aliasing, but will reduce some of the
                highest frequencies.

        Shape:

            - Input: `[*, T]`
            - Output: `[*, T']` with `T' = int(new_sr * T / old_sr)`


        .. caution::
            After dividing `old_sr` and `new_sr` by their GCD, both should be small
            for this implementation to be fast.

        >>> import paddle
        >>> resample = ResampleFrac(4, 5)
        >>> x = paddle.randn([1000])
        >>> print(len(resample(x)))
        1250
        """
        super().__init__()
        if not isinstance(old_sr, int) or not isinstance(new_sr, int):
            raise ValueError("old_sr and new_sr should be integers")
        gcd = math.gcd(old_sr, new_sr)
        self.old_sr = old_sr // gcd
        self.new_sr = new_sr // gcd
        self.zeros = zeros
        self.rolloff = rolloff

        self._init_kernels()

    def _init_kernels(self):
        if self.old_sr == self.new_sr:
            return

        kernels = []
        sr = min(self.new_sr, self.old_sr)
        sr *= self.rolloff

        self._width = math.ceil(self.zeros * self.old_sr / sr)
        idx = paddle.arange(
            -self._width, self._width + self.old_sr, dtype="float32")
        for i in range(self.new_sr):
            t = (-i / self.new_sr + idx / self.old_sr) * sr
            t = paddle.clip(t, -self.zeros, self.zeros)
            t *= math.pi
            window = paddle.cos(t / self.zeros / 2)**2
            kernel = sinc(t) * window
            # Renormalize kernel to ensure a constant signal is preserved.
            kernel = kernel / kernel.sum()
            kernels.append(kernel)

        _kernel = paddle.stack(kernels).reshape([self.new_sr, 1, -1])
        self.kernel = self.create_parameter(
            shape=_kernel.shape,
            dtype=_kernel.dtype, )
        self.kernel.set_value(_kernel)

    def forward(
            self,
            x: paddle.Tensor,
            output_length: Optional[int]=None,
            full: bool=False, ):
        """
        Resample x.
        Args:
            x (Tensor): signal to resample, time should be the last dimension
            output_length (None or int): This can be set to the desired output length
                (last dimension). Allowed values are between 0 and
                ceil(length * new_sr / old_sr). When None (default) is specified, the
                floored output length will be used. In order to select the largest possible
                size, use the `full` argument.
            full (bool): return the longest possible output from the input. This can be useful
                if you chain resampling operations, and want to give the `output_length` only
                for the last one, while passing `full=True` to all the other ones.
        """
        if self.old_sr == self.new_sr:
            return x
        shape = x.shape
        _dtype = x.dtype
        length = x.shape[-1]
        x = x.reshape([-1, length])
        x = F.pad(
            x.unsqueeze(1),
            [self._width, self._width + self.old_sr],
            mode="replicate",
            data_format="NCL", ).astype(self.kernel.dtype)
        ys = F.conv1d(x, self.kernel, stride=self.old_sr, data_format="NCL")
        y = ys.transpose(
            [0, 2, 1]).reshape(list(shape[:-1]) + [-1]).astype(_dtype)

        float_output_length = paddle.to_tensor(
            self.new_sr * length / self.old_sr, dtype="float32")
        max_output_length = paddle.ceil(float_output_length).astype("int64")
        default_output_length = paddle.floor(float_output_length).astype(
            "int64")

        if output_length is None:
            applied_output_length = (max_output_length
                                     if full else default_output_length)
        elif output_length < 0 or output_length > max_output_length:
            raise ValueError(
                f"output_length must be between 0 and {max_output_length.numpy()}"
            )
        else:
            applied_output_length = paddle.to_tensor(
                output_length, dtype="int64")
            if full:
                raise ValueError(
                    "You cannot pass both full=True and output_length")
        return y[..., :applied_output_length]

    def __repr__(self):
        return simple_repr(self)


def resample_frac(
        x: paddle.Tensor,
        old_sr: int,
        new_sr: int,
        zeros: int=24,
        rolloff: float=0.945,
        output_length: Optional[int]=None,
        full: bool=False, ):
    """
    Functional version of `ResampleFrac`, refer to its documentation for more information.

    ..warning::
        If you call repeatidly this functions with the same sample rates, then the
        resampling kernel will be recomputed everytime. For best performance, you should use
        and cache an instance of `ResampleFrac`.
    """
    return ResampleFrac(old_sr, new_sr, zeros, rolloff)(x, output_length, full)


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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
