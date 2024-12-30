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
import numpy as np
import paddle

# TODO: Complete type-hint and doc string.


def blackman_window(win_len, dtype=np.float32):
    arcs = np.pi * np.arange(win_len) / float(win_len)
    win = np.asarray(
        [0.42 - 0.5 * np.cos(2 * arc) + 0.08 * np.cos(4 * arc) for arc in arcs],
        dtype=dtype)
    return paddle.to_tensor(win)


def compute_amplitude(waveforms, lengths=None, amp_type="avg", scale="linear"):
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
        out = paddle.max(paddle.abs(waveforms), axis=1, keepdim=True)
    else:
        raise NotImplementedError

    if scale == "linear":
        return out
    elif scale == "dB":
        return paddle.clip(20 * paddle.log10(out), min=-80)
    else:
        raise NotImplementedError


def dB_to_amplitude(SNR):
    return 10**(SNR / 20)


def convolve1d(
        waveform,
        kernel,
        padding=0,
        pad_type="constant",
        stride=1,
        groups=1, ):
    if len(waveform.shape) != 3:
        raise ValueError("Convolve1D expects a 3-dimensional tensor")

    # Padding can be a tuple (left_pad, right_pad) or an int
    if isinstance(padding, list):
        waveform = paddle.nn.functional.pad(
            x=waveform,
            pad=padding,
            mode=pad_type,
            data_format='NLC', )

    # Move time dimension last, which pad and fft and conv expect.
    # (N, L, C) -> (N, C, L)
    waveform = waveform.transpose([0, 2, 1])
    kernel = kernel.transpose([0, 2, 1])

    convolved = paddle.nn.functional.conv1d(
        x=waveform,
        weight=kernel,
        stride=stride,
        groups=groups,
        padding=padding if not isinstance(padding, list) else 0, )

    # Return time dimension to the second dimension.
    return convolved.transpose([0, 2, 1])


def notch_filter(notch_freq, filter_width=101, notch_width=0.05):
    # Check inputs
    assert 0 < notch_freq <= 1
    assert filter_width % 2 != 0
    pad = filter_width // 2
    inputs = paddle.arange(filter_width, dtype='float32') - pad

    # Avoid frequencies that are too low
    notch_freq += notch_width

    # Define sinc function, avoiding division by zero
    def sinc(x):
        def _sinc(x):
            return paddle.sin(x) / x

        # The zero is at the middle index
        res = paddle.concat(
            [_sinc(x[:pad]), paddle.ones([1]), _sinc(x[pad + 1:])])
        return res

    # Compute a low-pass filter with cutoff frequency notch_freq.
    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    # import torch
    # hlpf *= paddle.to_tensor(torch.blackman_window(filter_width).detach().numpy())
    hlpf *= blackman_window(filter_width)
    hlpf /= paddle.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency notch_freq.
    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    # hhpf *= paddle.to_tensor(torch.blackman_window(filter_width).detach().numpy())
    hhpf *= blackman_window(filter_width)
    hhpf /= -paddle.sum(hhpf)
    hhpf[pad] += 1

    # Adding filters creates notch filter
    return (hlpf + hhpf).reshape([1, -1, 1])


def reverberate(waveforms,
                rir_waveform,
                sample_rate,
                impulse_duration=0.3,
                rescale_amp="avg"):
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
    orig_amplitude = compute_amplitude(waveforms, waveforms.shape[1],
                                       rescale_amp)

    # Compute index of the direct signal, so we can preserve alignment
    impulse_index_start = rir_waveform.abs().argmax(axis=1).item()
    impulse_index_end = min(
        impulse_index_start + int(sample_rate * impulse_duration),
        rir_waveform.shape[1])
    rir_waveform = rir_waveform[:, impulse_index_start:impulse_index_end, :]
    rir_waveform = rir_waveform / paddle.norm(rir_waveform, p=2)
    rir_waveform = paddle.flip(rir_waveform, [1])

    waveforms = convolve1d(
        waveform=waveforms,
        kernel=rir_waveform,
        padding=[rir_waveform.shape[1] - 1, 0], )

    # Rescale to the peak amplitude of the clean waveform
    waveforms = rescale(waveforms, waveforms.shape[1], orig_amplitude,
                        rescale_amp)

    if len(orig_shape) == 1:
        waveforms = waveforms.squeeze(0).squeeze(-1)
    if len(orig_shape) == 2:
        waveforms = waveforms.squeeze(-1)

    return waveforms


def rescale(waveforms, lengths, target_lvl, amp_type="avg", scale="linear"):
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


def normalize(waveforms, lengths=None, amp_type="avg", eps=1e-14):
    assert amp_type in ["avg", "peak"]

    batch_added = False
    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = waveforms.unsqueeze(0)

    den = compute_amplitude(waveforms, lengths, amp_type) + eps
    if batch_added:
        waveforms = waveforms.squeeze(0)
    return waveforms / den
