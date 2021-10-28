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

import paddleaudio as pa


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def load_audio():
    x, r = librosa.load('./test/data/test_audio.wav')
    #x,r = librosa.load('../data/test_audio.wav',sr=16000)
    return x, r


## start testing
x, r = load_audio()
EPS = 1e-8


def relative_err(a, b, real=True):
    """compute relative error of two matrices or vectors"""
    if real:
        return np.sum((a - b)**2) / (EPS + np.sum(a**2) + np.sum(b**2))
    else:
        err = np.sum((a.real - b.real)**2) / (
            EPS + np.sum(a.real**2) + np.sum(b.real**2))
        err += np.sum((a.imag - b.imag)**2) / (
            EPS + np.sum(a.imag**2) + np.sum(b.imag**2))

        return err


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_melspectrogram():
    a = pa.melspectrogram(
        x,
        window_size=512,
        sr=16000,
        hop_length=320,
        n_mels=64,
        fmin=50,
        to_db=False, )
    b = librosa.feature.melspectrogram(
        x,
        sr=16000,
        n_fft=512,
        win_length=512,
        hop_length=320,
        n_mels=64,
        fmin=50)
    assert relative_err(a, b) < EPS


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_melspectrogram_db():

    a = pa.melspectrogram(
        x,
        window_size=512,
        sr=16000,
        hop_length=320,
        n_mels=64,
        fmin=50,
        to_db=True,
        ref=1.0,
        amin=1e-10,
        top_db=None)
    b = librosa.feature.melspectrogram(
        x,
        sr=16000,
        n_fft=512,
        win_length=512,
        hop_length=320,
        n_mels=64,
        fmin=50)
    b = pa.power_to_db(b, ref=1.0, amin=1e-10, top_db=None)
    assert relative_err(a, b) < EPS


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_stft():
    a = pa.stft(x, n_fft=1024, hop_length=320, win_length=512)
    b = librosa.stft(x, n_fft=1024, hop_length=320, win_length=512)
    assert a.shape == b.shape
    assert relative_err(a, b, real=False) < EPS


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_split_frames():
    a = librosa.util.frame(x, frame_length=512, hop_length=320)
    b = pa.split_frames(x, frame_length=512, hop_length=320)
    assert relative_err(a, b) < EPS


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_mfcc():
    kwargs = {
        'window_size': 512,
        'hop_length': 320,
        'n_mels': 64,
        'fmin': 50,
        'to_db': False
    }
    a = pa.mfcc(
        x,
        #sample_rate=16000,
        spect=None,
        n_mfcc=20,
        dct_type=2,
        norm='ortho',
        lifter=0,
        **kwargs)
    S = librosa.feature.melspectrogram(
        x,
        sr=16000,
        n_fft=512,
        win_length=512,
        hop_length=320,
        n_mels=64,
        fmin=50)
    b = librosa.feature.mfcc(
        x, sr=16000, S=S, n_mfcc=20, dct_type=2, norm='ortho', lifter=0)
    assert relative_err(a, b) < EPS


if __name__ == '__main__':
    test_melspectrogram()
    test_melspectrogram_db()
    test_stft()
    test_split_frames()
    test_mfcc()
