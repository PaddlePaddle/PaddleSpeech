# Copyright (c) 2023 speechbrain Authors. All Rights Reserved.
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from speechbrain 2023 (https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/processing/signal_processing.py)
"""
Low level signal processing utilities
Authors
 * Peter Plantinga 2020
 * Francois Grondin 2020
 * William Aris 2020
 * Samuele Cornell 2020
 * Sarthak Yadav 2022
"""
import numpy as np
import paddle


def blackman_window(window_length, periodic=True):
    """Blackman window function.
    Arguments
    ---------
    window_length : int
        Controlling the returned window size. 
    periodic : bool
        Determines whether the returned window trims off the
        last duplicate value from the symmetric window

    Returns
    -------
    A 1-D tensor of size (window_length) containing the window
    """
    if window_length == 0:
        return []
    if window_length == 1:
        return paddle.ones([1])
    if periodic:
        window_length += 1
    window = paddle.arange(window_length) * (np.pi / (window_length - 1))
    window = 0.08 * paddle.cos(window * 4) - 0.5 * paddle.cos(window * 2) + 0.42
    return window[:-1] if periodic else window


def compute_amplitude(waveforms, lengths=None, amp_type="avg", scale="linear"):
    """Compute amplitude of a batch of waveforms.
    Arguments
    ---------
    waveform : tensor
        The waveforms used for computing amplitude.
        Shape should be `[time]` or `[batch, time]` or
        `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether to compute "avg" average or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        Whether to compute amplitude in "dB" or "linear" scale.
        Choose between ["linear", "dB"].
    Returns
    -------
    The average amplitude of the waveforms.
    Example
    -------
    >>> signal = paddle.sin(paddle.arange(16000.0)).unsqueeze(0)
    >>> compute_amplitude(signal, signal.size(1))
    tensor([[0.6366]])
    """
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0)

    assert amp_type in ["avg", "peak"]
    assert scale in ["linear", "dB"]

    if amp_type == "avg":
        if lengths is None:
            out = paddle.mean(paddle.abs(waveforms), axis=1, keepdim=True)
        else:
            wav_sum = paddle.sum(paddle.abs(waveforms), axis=1, keepdim=True)
            out = wav_sum / lengths.astype(wav_sum.dtype)
    elif amp_type == "peak":
        out = paddle.max(paddle.abs(waveforms), axis=1, keepdim=True)[0]
    else:
        raise NotImplementedError

    if scale == "linear":
        return out
    elif scale == "dB":
        return paddle.clip(20 * paddle.log10(out), min=-80)  # clamp zeros
    else:
        raise NotImplementedError


def convolve1d(
        waveform,
        kernel,
        padding=0,
        pad_type="constant",
        stride=1,
        groups=1,
        use_fft=False,
        rotation_index=0, ):
    """Use paddle.nn.functional to perform 1d padding and conv.
    Arguments
    ---------
    waveform : tensor
        The tensor to perform operations on.
    kernel : tensor
        The filter to apply during convolution.
    padding : int or tuple
        The padding (pad_left, pad_right) to apply.
        If an integer is passed instead, this is passed
        to the conv1d function and pad_type is ignored.
    pad_type : str
        The type of padding to use. Passed directly to
        `paddle.nn.functional.pad`, see Paddle documentation
        for available options.
    stride : int
        The number of units to move each time convolution is applied.
        Passed to conv1d. Has no effect if `use_fft` is True.
    groups : int
        This option is passed to `conv1d` to split the input into groups for
        convolution. Input channels should be divisible by the number of groups.
    use_fft : bool
        When `use_fft` is passed `True`, then compute the convolution in the
        spectral domain using complex multiply. This is more efficient on CPU
        when the size of the kernel is large (e.g. reverberation). WARNING:
        Without padding, circular convolution occurs. This makes little
        difference in the case of reverberation, but may make more difference
        with different kernels.
    rotation_index : int
        This option only applies if `use_fft` is true. If so, the kernel is
        rolled by this amount before convolution to shift the output location.
    Returns
    -------
    The convolved waveform.
    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> signal = signal.unsqueeze(0).unsqueeze(2)
    >>> kernel = paddle.rand([1, 10, 1])
    >>> signal = convolve1d(signal, kernel, padding=(9, 0))
    """
    if len(waveform.shape) != 3:
        raise ValueError("Convolve1D expects a 3-dimensional tensor")

    # Move time dimension last, which pad and fft and conv expect.
    waveform = waveform.transpose([0, 2, 1])
    kernel = kernel.transpose([0, 2, 1])
    # Padding can be a tuple (left_pad, right_pad) or an int
    if isinstance(padding, tuple):
        waveform = paddle.nn.functional.pad(
            x=waveform, pad=padding, mode=pad_type, data_format='NCL')

    # This approach uses FFT, which is more efficient if the kernel is large
    if use_fft:
        # Pad kernel to same length as signal, ensuring correct alignment
        zero_length = waveform.shape[-1] - kernel.shape[-1]

        # Handle case where signal is shorter
        if zero_length < 0:
            kernel = kernel[..., :zero_length]
            zero_length = 0

        # Perform rotation to ensure alignment
        zeros = paddle.zeros(
            [kernel.shape[0], kernel.shape[1], zero_length], dtype=kernel.dtype)
        after_index = kernel[..., rotation_index:]
        before_index = kernel[..., :rotation_index]
        kernel = paddle.concat((after_index, zeros, before_index), axis=-1)

        # Multiply in frequency domain to convolve in time domain
        import paddle.fft as fft

        result = fft.rfft(waveform) * fft.rfft(kernel)
        convolved = fft.irfft(result, n=waveform.shape[-1])

    # Use the implementation given by paddle, which should be efficient on GPU
    else:
        convolved = paddle.nn.functional.conv1d(
            x=waveform,
            weight=kernel,
            stride=stride,
            groups=groups,
            padding=padding if not isinstance(padding, tuple) else 0, )

    # Return time dimension to the second dimension.
    return convolved.transpose([0, 2, 1])


def notch_filter(notch_freq, filter_width=101, notch_width=0.05):
    """Returns a notch filter constructed from a high-pass and low-pass filter.
    (from https://tomroelandts.com/articles/
    how-to-create-simple-band-pass-and-band-reject-filters)
    Arguments
    ---------
    notch_freq : float
        frequency to put notch as a fraction of the
        sampling rate / 2. The range of possible inputs is 0 to 1.
    filter_width : int
        Filter width in samples. Longer filters have
        smaller transition bands, but are more inefficient.
    notch_width : float
        Width of the notch, as a fraction of the sampling_rate / 2.
    """

    # Check inputs
    assert 0 < notch_freq <= 1
    assert filter_width % 2 != 0
    pad = filter_width // 2
    inputs = paddle.arange(filter_width) - pad

    # Avoid frequencies that are too low
    notch_freq += notch_width

    # Define sinc function, avoiding division by zero
    def sinc(x):
        "Computes the sinc function."

        def _sinc(x):
            return paddle.sin(x) / x

        # The zero is at the middle index
        return paddle.concat(
            [_sinc(x[:pad]), paddle.ones([1]), _sinc(x[pad + 1:])])

    # Compute a low-pass filter with cutoff frequency notch_freq.
    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    hlpf *= blackman_window(filter_width)
    hlpf /= paddle.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency notch_freq.
    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    hhpf *= blackman_window(filter_width)
    hhpf /= -paddle.sum(hhpf)
    hhpf[pad] += 1

    # Adding filters creates notch filter
    return (hlpf + hhpf).reshape([1, -1, 1])
