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
import librosa
import numpy as np
import pytest

import paddleaudio

TEST_FILE = './test/data/test_audio.wav'


def relative_err(a, b, real=True):
    """compute relative error of two matrices or vectors"""
    if real:
        return np.sum((a - b)**2) / (EPS + np.sum(a**2) + np.sum(b**2))
    else:
        err = np.sum((a.real - b.real)**2) / \
            (EPS + np.sum(a.real**2) + np.sum(b.real**2))
        err += np.sum((a.imag - b.imag)**2) / \
            (EPS + np.sum(a.imag**2) + np.sum(b.imag**2))

        return err


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def load_audio():
    x, r = librosa.load(TEST_FILE, sr=16000)
    print(f'librosa: mean: {np.mean(x)}, std:{np.std(x)}')
    return x, r


# start testing
x, r = load_audio()
EPS = 1e-8


def test_load():
    s, r = paddleaudio.load(TEST_FILE, sr=16000)
    assert r == 16000
    assert s.dtype == 'float32'

    s, r = paddleaudio.load(
        TEST_FILE, sr=16000, offset=1, duration=2, dtype='int16')
    assert len(s) / r == 2.0
    assert r == 16000
    assert s.dtype == 'int16'


def test_depth_convert():
    y = paddleaudio.depth_convert(x, 'int16')
    assert len(y) == len(x)
    assert y.dtype == 'int16'
    assert np.max(y) <= 32767
    assert np.min(y) >= -32768
    assert np.std(y) > EPS

    y = paddleaudio.depth_convert(x, 'int8')
    assert len(y) == len(x)
    assert y.dtype == 'int8'
    assert np.max(y) <= 127
    assert np.min(y) >= -128
    assert np.std(y) > EPS


# test case for resample
rs_test_data = [
    (32000, 'kaiser_fast'),
    (16000, 'kaiser_fast'),
    (8000, 'kaiser_fast'),
    (32000, 'kaiser_best'),
    (16000, 'kaiser_best'),
    (8000, 'kaiser_best'),
    (22050, 'kaiser_best'),
    (44100, 'kaiser_best'),
]


@pytest.mark.parametrize('sr,mode', rs_test_data)
def test_resample(sr, mode):
    y = paddleaudio.resample(x, 16000, sr, mode=mode)
    factor = sr / 16000
    err = relative_err(len(y), len(x) * factor)
    print('err:', err)
    assert err < EPS


def test_normalize():
    y = paddleaudio.normalize(x, norm_type='linear', mul_factor=0.5)
    assert np.max(y) < 0.5 + EPS

    y = paddleaudio.normalize(x, norm_type='linear', mul_factor=2.0)
    assert np.max(y) <= 2.0 + EPS

    y = paddleaudio.normalize(x, norm_type='gaussian', mul_factor=1.0)
    print('np.std(y):', np.std(y))
    assert np.abs(np.std(y) - 1.0) < EPS


if __name__ == '__main__':
    test_load()
    test_depth_convert()
    test_resample(22050, 'kaiser_fast')
    test_normalize()
