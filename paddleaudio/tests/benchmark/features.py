# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle

import paddleaudio

# Feature conf
mel_conf = {
    'sr': 16000,
    'n_fft': 512,
    'hop_length': 128,
    'n_mels': 40,
}
mfcc_conf = {
    'n_mfcc': 20,
    'top_db': 80.0,
}
mfcc_conf.update(mel_conf)

input_shape = (48000)
waveform = np.random.random(size=input_shape)
waveform_tensor = paddle.to_tensor(waveform).unsqueeze(0)


def enable_cpu_device():
    paddle.set_device('cpu')


def enable_gpu_device():
    paddle.set_device('gpu')


mel_extractor = paddleaudio.features.MelSpectrogram(
    **mel_conf, f_min=0.0, dtype=waveform_tensor.dtype)


def melspectrogram():
    return mel_extractor(waveform_tensor).squeeze(0)


def test_melspect_cpu(benchmark):
    enable_cpu_device()
    feature_paddleaudio = benchmark(melspectrogram)
    feature_librosa = librosa.feature.melspectrogram(waveform, **mel_conf)
    np.testing.assert_array_almost_equal(
        feature_librosa, feature_paddleaudio, decimal=4)


def test_melspect_gpu(benchmark):
    enable_gpu_device()
    feature_paddleaudio = benchmark(melspectrogram)
    feature_librosa = librosa.feature.melspectrogram(waveform, **mel_conf)
    np.testing.assert_array_almost_equal(
        feature_librosa, feature_paddleaudio, decimal=4)


log_mel_extractor = paddleaudio.features.LogMelSpectrogram(
    **mel_conf, f_min=0.0, dtype=waveform_tensor.dtype)


def log_melspectrogram():
    return log_mel_extractor(waveform_tensor).squeeze(0)


def test_log_melspect_cpu(benchmark):
    enable_cpu_device()
    feature_paddleaudio = benchmark(log_melspectrogram)
    feature_librosa = librosa.feature.melspectrogram(waveform, **mel_conf)
    feature_librosa = librosa.power_to_db(feature_librosa, top_db=None)
    np.testing.assert_array_almost_equal(
        feature_librosa, feature_paddleaudio, decimal=4)


def test_log_melspect_gpu(benchmark):
    enable_gpu_device()
    feature_paddleaudio = benchmark(log_melspectrogram)
    feature_librosa = librosa.feature.melspectrogram(waveform, **mel_conf)
    feature_librosa = librosa.power_to_db(feature_librosa, top_db=None)
    np.testing.assert_array_almost_equal(
        feature_librosa, feature_paddleaudio, decimal=4)


mfcc_extractor = paddleaudio.features.MFCC(
    **mfcc_conf, f_min=0.0, dtype=waveform_tensor.dtype)


def mfcc():
    return mfcc_extractor(waveform_tensor).squeeze(0)


def test_mfcc_cpu(benchmark):
    enable_cpu_device()
    feature_paddleaudio = benchmark(mfcc)
    feature_librosa = librosa.feature.mfcc(waveform, **mel_conf)
    np.testing.assert_array_almost_equal(
        feature_librosa, feature_paddleaudio, decimal=4)


def test_mfcc_gpu(benchmark):
    enable_gpu_device()
    feature_paddleaudio = benchmark(mfcc)
    feature_librosa = librosa.feature.mfcc(waveform, **mel_conf)
    np.testing.assert_array_almost_equal(
        feature_librosa, feature_paddleaudio, decimal=4)
