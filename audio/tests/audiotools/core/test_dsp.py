# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/core/test_dsp.py)
import sys

import numpy as np
import paddle
import pytest

from audio.audiotools import AudioSignal
from audio.audiotools.core.util import sample_from_dist


@pytest.mark.parametrize("window_duration", [0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100])
@pytest.mark.parametrize("duration", [0.5, 1.0, 2.0, 10.0])
def test_overlap_add(duration, sample_rate, window_duration):
    np.random.seed(0)
    if duration > window_duration:
        spk_signal = AudioSignal.batch([
            AudioSignal.excerpt(
                "./audio/spk/f10_script4_produced.wav", duration=duration)
            for _ in range(16)
        ])
        spk_signal.resample(sample_rate)

        noise = paddle.randn([16, 1, int(duration * sample_rate)])
        nz_signal = AudioSignal(noise, sample_rate=sample_rate)

        def _test(signal):
            hop_duration = window_duration / 2
            windowed_signal = signal.clone().collect_windows(window_duration,
                                                             hop_duration)
            recombined = windowed_signal.overlap_and_add(hop_duration)

            assert recombined == signal
            assert np.allclose(recombined.audio_data, signal.audio_data, 1e-3)

        _test(nz_signal)
        _test(spk_signal)


@pytest.mark.parametrize("window_duration", [0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100])
@pytest.mark.parametrize("duration", [0.5, 1.0, 2.0, 10.0])
def test_inplace_overlap_add(duration, sample_rate, window_duration):
    np.random.seed(0)
    if duration > window_duration:
        spk_signal = AudioSignal.batch([
            AudioSignal.excerpt(
                "./audio/spk/f10_script4_produced.wav", duration=duration)
            for _ in range(16)
        ])
        spk_signal.resample(sample_rate)

        noise = paddle.randn([16, 1, int(duration * sample_rate)])
        nz_signal = AudioSignal(noise, sample_rate=sample_rate)

        def _test(signal):
            hop_duration = window_duration / 2
            windowed_signal = signal.clone().collect_windows(window_duration,
                                                             hop_duration)
            # Compare in-place with unfold results
            for i, window in enumerate(
                    signal.clone().windows(window_duration, hop_duration)):
                assert np.allclose(window.audio_data,
                                   windowed_signal.audio_data[i])

        _test(nz_signal)
        _test(spk_signal)


def test_low_pass():
    sample_rate = 44100
    f = 440
    t = paddle.arange(0, 1, 1 / sample_rate)
    sine_wave = paddle.sin(2 * np.pi * f * t)
    window = AudioSignal.get_window("hann", sine_wave.shape[-1])
    sine_wave = sine_wave * window
    signal = AudioSignal(sine_wave.unsqueeze(0), sample_rate=sample_rate)
    out = signal.clone().low_pass(220)
    assert out.audio_data.abs().max() < 1e-4

    out = signal.clone().low_pass(880)
    assert (out - signal).audio_data.abs().max() < 1e-3

    batch = AudioSignal.batch([signal.clone(), signal.clone(), signal.clone()])

    cutoffs = [220, 880, 220]
    out = batch.clone().low_pass(cutoffs)

    assert out.audio_data[0].abs().max() < 1e-4
    assert out.audio_data[2].abs().max() < 1e-4
    assert (out - batch).audio_data[1].abs().max() < 1e-3


def test_high_pass():
    sample_rate = 44100
    f = 440
    t = paddle.arange(0, 1, 1 / sample_rate)
    sine_wave = paddle.sin(2 * np.pi * f * t)
    window = AudioSignal.get_window("hann", sine_wave.shape[-1])
    sine_wave = sine_wave * window
    signal = AudioSignal(sine_wave.unsqueeze(0), sample_rate=sample_rate)
    out = signal.clone().high_pass(220)
    assert (signal - out).audio_data.abs().max() < 1e-4


def test_mask_frequencies():
    sample_rate = 44100
    fs = paddle.to_tensor([500.0, 2000.0, 8000.0, 32000.0])[None]
    t = paddle.arange(0, 1, 1 / sample_rate)[:, None]
    sine_wave = paddle.sin(2 * np.pi * t @ fs).sum(axis=-1)
    sine_wave = AudioSignal(sine_wave, sample_rate)
    masked_sine_wave = sine_wave.mask_frequencies(fmin_hz=1500, fmax_hz=10000)

    fs2 = paddle.to_tensor([500.0, 32000.0])[None]
    sine_wave2 = paddle.sin(2 * np.pi * t @ fs).sum(axis=-1)
    sine_wave2 = AudioSignal(sine_wave2, sample_rate)

    assert paddle.allclose(masked_sine_wave.audio_data, sine_wave2.audio_data)


def test_mask_timesteps():
    sample_rate = 44100
    f = 440
    t = paddle.linspace(0, 1, sample_rate)
    sine_wave = paddle.sin(2 * np.pi * f * t)
    sine_wave = AudioSignal(sine_wave, sample_rate)

    masked_sine_wave = sine_wave.mask_timesteps(tmin_s=0.25, tmax_s=0.75)
    masked_sine_wave.istft()

    mask = ((0.3 < t) & (t < 0.7))[None, None]
    assert paddle.allclose(
        masked_sine_wave.audio_data[mask],
        paddle.zeros_like(masked_sine_wave.audio_data[mask]), )


def test_shift_phase():
    sample_rate = 44100
    f = 440
    t = paddle.linspace(0, 1, sample_rate)
    sine_wave = paddle.sin(2 * np.pi * f * t)
    sine_wave = AudioSignal(sine_wave, sample_rate)
    sine_wave2 = sine_wave.clone()

    shifted_sine_wave = sine_wave.shift_phase(np.pi)
    shifted_sine_wave.istft()

    sine_wave2.phase = sine_wave2.phase + np.pi
    sine_wave2.istft()

    assert paddle.allclose(shifted_sine_wave.audio_data, sine_wave2.audio_data)


def test_corrupt_phase():
    sample_rate = 44100
    f = 440
    t = paddle.linspace(0, 1, sample_rate)
    sine_wave = paddle.sin(2 * np.pi * f * t)
    sine_wave = AudioSignal(sine_wave, sample_rate)
    sine_wave2 = sine_wave.clone()

    shifted_sine_wave = sine_wave.corrupt_phase(scale=np.pi)
    shifted_sine_wave.istft()

    assert (sine_wave2.phase - shifted_sine_wave.phase).abs().mean() > 0.0
    assert ((sine_wave2.phase - shifted_sine_wave.phase).std() / np.pi) < 1.0


def test_preemphasis():
    x = AudioSignal.excerpt("./audio/spk/f10_script4_produced.wav", duration=5)
    import matplotlib.pyplot as plt

    x.specshow(preemphasis=False)

    x.specshow(preemphasis=True)

    x.preemphasis()
