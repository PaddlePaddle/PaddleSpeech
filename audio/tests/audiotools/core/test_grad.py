# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#
# This file contains code derived from [Descript]
# (https://github.com/descriptinc/audiotools),
# which is licensed under the MIT License:
#
# MIT License
#
# Copyright (c) 2023-Present, Descript
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import sys
from typing import Callable

import numpy as np
import paddle
import pytest
sys.path.append("../..")
from audiotools import AudioSignal


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
