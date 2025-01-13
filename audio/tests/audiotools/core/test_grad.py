# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/core/test_grad.py)
import sys
from typing import Callable

import numpy as np
import paddle
import pytest

from audio.audiotools import AudioSignal


def test_audio_grad():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    ir_path = "./audio/ir/h179_Bar_1txts.wav"

    def _test_audio_grad(attr: str, target=True, kwargs: dict={}):
        signal = AudioSignal(audio_path)
        signal.audio_data.stop_gradient = False

        assert signal.audio_data.grad is None

        # Avoid overwriting leaf tensor by cloning signal
        attr = getattr(signal.clone(), attr)
        result = attr(**kwargs) if isinstance(attr, Callable) else attr

        try:
            if isinstance(result, AudioSignal):
                # If necessary, propagate spectrogram changes to waveform
                if result.stft_data is not None:
                    result.istft()
                # if result.audio_data.dtype.is_complex:
                if paddle.is_complex(result.audio_data):
                    result.audio_data.real.sum().backward()
                else:
                    result.audio_data.sum().backward()
            else:
                # if result.dtype.is_complex:
                if paddle.is_complex(result):
                    result.real().sum().backward()
                else:
                    result.sum().backward()

            assert signal.audio_data.grad is not None or not target
        except RuntimeError:
            assert not target

    for a in [
        ["mix", True, {
            "other": AudioSignal(audio_path),
            "snr": 0
        }],
        ["convolve", True, {
            "other": AudioSignal(ir_path)
        }],
        [
            "apply_ir",
            True,
            {
                "ir": AudioSignal(ir_path),
                "drr": 0.1,
                "ir_eq": paddle.randn([6])
            },
        ],
        ["ensure_max_of_audio", True],
        ["normalize", True],
        ["volume_change", True, {
            "db": 1
        }],
            # ["pitch_shift", False, {"n_semitones": 1}],
            # ["time_stretch", False, {"factor": 2}],
            # ["apply_codec", False],
        ["equalizer", True, {
            "db": paddle.randn([6])
        }],
        ["clip_distortion", True, {
            "clip_percentile": 0.5
        }],
        ["quantization", True, {
            "quantization_channels": 8
        }],
        ["mulaw_quantization", True, {
            "quantization_channels": 8
        }],
        ["resample", True, {
            "sample_rate": 16000
        }],
        ["low_pass", True, {
            "cutoffs": 1000
        }],
        ["high_pass", True, {
            "cutoffs": 1000
        }],
        ["to_mono", True],
        ["zero_pad", True, {
            "before": 10,
            "after": 10
        }],
        ["magnitude", True],
        ["phase", True],
        ["log_magnitude", True],
        ["loudness", False],
        ["stft", True],
        ["clone", True],
        ["mel_spectrogram", True],
        ["zero_pad_to", True, {
            "length": 100000
        }],
        ["truncate_samples", True, {
            "length_in_samples": 1000
        }],
        ["corrupt_phase", True, {
            "scale": 0.5
        }],
        ["shift_phase", True, {
            "shift": 1
        }],
        ["mask_low_magnitudes", True, {
            "db_cutoff": 0
        }],
        ["mask_frequencies", True, {
            "fmin_hz": 100,
            "fmax_hz": 1000
        }],
        ["mask_timesteps", True, {
            "tmin_s": 0.1,
            "tmax_s": 0.5
        }],
        ["__add__", True, {
            "other": AudioSignal(audio_path)
        }],
        ["__iadd__", True, {
            "other": AudioSignal(audio_path)
        }],
        ["__radd__", True, {
            "other": AudioSignal(audio_path)
        }],
        ["__sub__", True, {
            "other": AudioSignal(audio_path)
        }],
        ["__isub__", True, {
            "other": AudioSignal(audio_path)
        }],
        ["__mul__", True, {
            "other": AudioSignal(audio_path)
        }],
        ["__imul__", True, {
            "other": AudioSignal(audio_path)
        }],
        ["__rmul__", True, {
            "other": AudioSignal(audio_path)
        }],
    ]:
        _test_audio_grad(*a)


def test_batch_grad():
    audio_path = "./audio/spk/f10_script4_produced.wav"

    signal = AudioSignal(audio_path)
    signal.audio_data.stop_gradient = False

    assert signal.audio_data.grad is None

    batch_size = 16
    batch = AudioSignal.batch([signal.clone() for _ in range(batch_size)])

    batch.audio_data.sum().backward()

    assert signal.audio_data.grad is not None
