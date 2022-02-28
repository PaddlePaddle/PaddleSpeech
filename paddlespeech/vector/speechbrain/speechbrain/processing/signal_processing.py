"""
Low level signal processing utilities

Authors
 * Peter Plantinga 2020
 * Francois Grondin 2020
 * William Aris 2020
 * Samuele Cornell 2020
"""
import paddle
import math
import numpy as np
from packaging import version


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
            wav_sum = paddle.sum(x=paddle.abs(waveforms), axis=1, keepdim=True)
            out = wav_sum / lengths
    elif amp_type == "peak":
        out = paddle.max(paddle.abs(waveforms))[0]
    else:
        raise NotImplementedError

    if scale == "linear":
        return out
    elif scale == "dB":
        return paddle.clip(20 * paddle.log10(out), min=-80)  # clamp zeros
    else:
        raise NotImplementedError


def normalize(waveforms, lengths=None, amp_type="avg", eps=1e-14):
    """This function normalizes a signal to unitary average or peak amplitude.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether one wants to normalize with respect to "avg" or "peak"
        amplitude. Choose between ["avg", "peak"]. Note: for "avg" clipping
        is not prevented and can occur.
    eps : float
        A small number to add to the denominator to prevent NaN.

    Returns
    -------
    waveforms : tensor
        Normalized level waveform.
    """

    assert amp_type in ["avg", "peak"]

    batch_added = False
    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = waveforms.unsqueeze(0)

    den = compute_amplitude(waveforms, lengths, amp_type) + eps
    if batch_added:
        waveforms = waveforms.squeeze(0)
    return waveforms / den


def rescale(waveforms, lengths, target_lvl, amp_type="avg", scale="linear"):
    """This functions performs signal rescaling to a target level.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    target_lvl : float
        Target lvl in dB or linear scale.
    amp_type : str
        Whether one wants to rescale with respect to "avg" or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        whether target_lvl belongs to linear or dB scale.
        Choose between ["linear", "dB"].

    Returns
    -------
    waveforms : tensor
        Rescaled waveforms.
    """

    assert amp_type in ["peak", "avg"]
    assert scale in ["linear", "dB"]

    batch_added = False
    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = waveforms.unsqueeze(0)

    waveforms = normalize(waveforms, lengths, amp_type)

    if scale == "linear":
        out = target_lvl * waveforms
    elif scale == "dB":
        out = dB_to_amplitude(target_lvl) * waveforms

    else:
        raise NotImplementedError("Invalid scale, choose between dB and linear")

    if batch_added:
        out = out.squeeze(0)

    return out


def convolve1d(
    waveform,
    kernel,
    padding=0,
    pad_type="constant",
    stride=1,
    groups=1,
    use_fft=False,
    rotation_index=0,
):
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
        `paddle.nn.functional.pad`, see PyTorch documentation
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
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> signal = signal.unsqueeze(0).unsqueeze(2)
    >>> kernel = paddle.rand(1, 10, 1)
    >>> signal = convolve1d(signal, kernel, padding=(9, 0))
    """
    if len(waveform.shape) != 3:
        raise ValueError("Convolve1D expects a 3-dimensional tensor")

    # Move time dimension last, which pad and fft and conv expect.
    # print("waveform shape: {}".format(waveform.shape))
    waveform = waveform.transpose(perm=[0,2, 1])
    kernel = kernel.transpose(perm=[0, 2, 1])

    # Padding can be a tuple (left_pad, right_pad) or an int
    if isinstance(padding, tuple):
        waveform = paddle.nn.functional.pad(
            input=waveform, pad=padding, mode=pad_type,
        )

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
            [kernel.shape[0], kernel.shape[1], zero_length], dtype="float32"
        )
        rotation_index = rotation_index.item()
        # print("rotation_index index: {}".format(rotation_index))
        # print("kernel index: {}".format(kernel))
        after_index = kernel[..., rotation_index:]
        # 这里当 rotation_index = 0的时候，无法取得元素
        if rotation_index == 0:
            before_index_shape = kernel.shape
            before_index_shape[-1] = 0
            before_index = paddle.zeros(shape=before_index_shape)
        else:
            before_index = kernel[..., :rotation_index]
        kernel = paddle.concat((after_index, zeros, before_index), axis=-1)

        # Multiply in frequency domain to convolve in time domain
        # if version.parse(paddle.__version__) > version.parse("1.6.0"):
        import paddle.fft as fft

        result = fft.rfft(waveform) * fft.rfft(kernel)
        convolved = fft.irfft(result, n=waveform.shape[-1])
        # else:
        #     f_signal = paddle.rfft(waveform, 1)
        #     f_kernel = paddle.rfft(kernel, 1)
        #     sig_real, sig_imag = f_signal.unbind(-1)
        #     ker_real, ker_imag = f_kernel.unbind(-1)
        #     f_result = paddle.stack(
        #         [
        #             sig_real * ker_real - sig_imag * ker_imag,
        #             sig_real * ker_imag + sig_imag * ker_real,
        #         ],
        #         dim=-1,
        #     )
        #     convolved = paddle.irfft(
        #         f_result, 1, signal_sizes=[waveform.size(-1)]
        #     )

    # Use the implementation given by paddle, which should be efficient on GPU
    else:
        convolved = paddle.nn.functional.conv1d(
            x=waveform,
            weight=kernel,
            stride=stride,
            groups=groups,
            padding=padding if not isinstance(padding, tuple) else 0,
        )

    # Return time dimension to the second dimension.
    return convolved.transpose(perm=[0,2, 1])


def reverberate(waveforms, rir_waveform, rescale_amp="avg"):
    """
    General function to contaminate a given signal with reverberation given a
    Room Impulse Response (RIR).
    It performs convolution between RIR and signal, but without changing
    the original amplitude of the signal.

    Arguments
    ---------
    waveforms : tensor
        The waveforms to normalize.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    rir_waveform : tensor
        RIR tensor, shape should be [time, channels].
    rescale_amp : str
        Whether reverberated signal is rescaled (None) and with respect either
        to original signal "peak" amplitude or "avg" average amplitude.
        Choose between [None, "avg", "peak"].

    Returns
    -------
    waveforms: tensor
        Reverberated signal.

    """

    orig_shape = waveforms.shape

    if len(waveforms.shape) > 3 or len(rir_waveform.shape) > 3:
        raise NotImplementedError

    # if inputs are mono tensors we reshape to 1, samples
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0).unsqueeze(-1)
    elif len(waveforms.shape) == 2:
        waveforms = waveforms.unsqueeze(-1)

    if len(rir_waveform.shape) == 1:  # convolve1d expects a 3d tensor !
        rir_waveform = rir_waveform.unsqueeze(0).unsqueeze(-1)
    elif len(rir_waveform.shape) == 2:
        rir_waveform = rir_waveform.unsqueeze(-1)

    # Compute the average amplitude of the clean
    orig_amplitude = compute_amplitude(
        waveforms, waveforms.shape[1], rescale_amp
    )

    # Compute index of the direct signal, so we can preserve alignment
    value_max = rir_waveform.abs().max(axis=1, keepdim=True)
    direct_index = rir_waveform.abs().argmax(axis=1, keepdim=True)
    # Making sure the max is always positive (if not, flip)
    # mask = paddle.logical_and(rir_waveform == value_max,  rir_waveform < 0)
    # rir_waveform[mask] = -rir_waveform[mask]

    # Use FFT to compute convolution, because of long reverberation filter
    waveforms = convolve1d(
        waveform=waveforms,
        kernel=rir_waveform,
        use_fft=True,
        rotation_index=direct_index,
    )

    # Rescale to the peak amplitude of the clean waveform
    waveforms = rescale(
        waveforms, waveforms.shape[1], orig_amplitude, rescale_amp
    )

    if len(orig_shape) == 1:
        waveforms = waveforms.squeeze(0).squeeze(-1)
    if len(orig_shape) == 2:
        waveforms = waveforms.squeeze(-1)

    return waveforms


def dB_to_amplitude(SNR):
    """Returns the amplitude ratio, converted from decibels.

    Arguments
    ---------
    SNR : float
        The ratio in decibels to convert.

    Example
    -------
    >>> round(dB_to_amplitude(SNR=10), 3)
    3.162
    >>> dB_to_amplitude(SNR=0)
    1.0
    """
    return 10 ** (SNR / 20)


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

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> signal = signal.unsqueeze(0).unsqueeze(2)
    >>> kernel = notch_filter(0.25)
    >>> notched_signal = convolve1d(signal, kernel)
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
        def _sinc(x):
            return paddle.sin(x) / x

        # The zero is at the middle index
        return paddle.concat([_sinc(x[:pad]), paddle.ones([1]), _sinc(x[pad + 1 :])])

    # Compute a low-pass filter with cutoff frequency notch_freq.
    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    # print("blackman: {}".format(np.blackman(filter_width)))
    hlpf *= paddle.to_tensor(np.blackman(filter_width))
    hlpf /= paddle.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency notch_freq.
    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    # print("blackman: {}".format(np.blackman(filter_width)))
    hhpf *= paddle.to_tensor(np.blackman(filter_width))
    hhpf /= -paddle.sum(hhpf)
    hhpf[pad] += 1

    # Adding filters creates notch filter
    return (hlpf + hhpf).reshape([1, -1, 1])


def overlap_and_add(signal, frame_step):
    """Taken from https://github.com/kaituoxu/Conv-TasNet/blob/master/src/utils.py

    Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py

    Example
    -------
    >>> signal = paddle.randn(5, 20)
    >>> overlapped = overlap_and_add(signal, 20)
    >>> overlapped.shape
    paddle.Size([100])
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(
        frame_length, frame_step
    )  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = paddle.arange(0, output_subframes).unfold(
        0, subframes_per_frame, subframe_step
    )

    # frame_old = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.clone().detach().to(signal.device.type)
    # print((frame - frame_old).sum())
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(
        *outer_dimensions, output_subframes, subframe_length
    )
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


def resynthesize(enhanced_mag, noisy_inputs, stft, istft, normalize_wavs=True):
    """Function for resynthesizing waveforms from enhanced mags.

    Arguments
    ---------
    enhanced_mag : paddle.Tensor
        Predicted spectral magnitude, should be three dimensional.
    noisy_inputs : paddle.Tensor
        The noisy waveforms before any processing, to extract phase.
    lengths : paddle.Tensor
        The length of each waveform for normalization.
    stft : paddle.nn.Layer
        Module for computing the STFT for extracting phase.
    istft : paddle.nn.Layer
        Module for computing the iSTFT for resynthesis.
    normalize_wavs : bool
        Whether to normalize the output wavs before returning them.

    Returns
    -------
    enhanced_wav : paddle.Tensor
        The resynthesized waveforms of the enhanced magnitudes with noisy phase.
    """

    # Extract noisy phase from inputs
    noisy_feats = stft(noisy_inputs)
    noisy_phase = paddle.atan2(noisy_feats[:, :, :, 1], noisy_feats[:, :, :, 0])

    # Combine with enhanced magnitude
    complex_predictions = paddle.mul(
        paddle.unsqueeze(enhanced_mag, -1),
        paddle.cat(
            (
                paddle.unsqueeze(paddle.cos(noisy_phase), -1),
                paddle.unsqueeze(paddle.sin(noisy_phase), -1),
            ),
            -1,
        ),
    )
    pred_wavs = istft(complex_predictions, sig_length=noisy_inputs.shape[1])

    # Normalize. Since we're using peak amplitudes, ignore lengths
    if normalize_wavs:
        pred_wavs = normalize(pred_wavs, amp_type="peak")

    return pred_wavs
