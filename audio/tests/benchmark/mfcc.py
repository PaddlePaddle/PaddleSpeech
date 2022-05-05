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
import os
import urllib.request

import librosa
import numpy as np
import paddle
import paddleaudio
import torch
import torchaudio

wav_url = 'https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav'
if not os.path.isfile(os.path.basename(wav_url)):
    urllib.request.urlretrieve(wav_url, os.path.basename(wav_url))

waveform, sr = paddleaudio.load(os.path.abspath(os.path.basename(wav_url)))
waveform_tensor = paddle.to_tensor(waveform).unsqueeze(0)
waveform_tensor_torch = torch.from_numpy(waveform).unsqueeze(0)

# Feature conf
mel_conf = {
    'sr': sr,
    'n_fft': 512,
    'hop_length': 128,
    'n_mels': 40,
}
mfcc_conf = {
    'n_mfcc': 20,
    'top_db': 80.0,
}
mfcc_conf.update(mel_conf)

mel_conf_torchaudio = {
    'sample_rate': sr,
    'n_fft': 512,
    'hop_length': 128,
    'n_mels': 40,
    'norm': 'slaney',
    'mel_scale': 'slaney',
}
mfcc_conf_torchaudio = {
    'sample_rate': sr,
    'n_mfcc': 20,
}


def enable_cpu_device():
    paddle.set_device('cpu')


def enable_gpu_device():
    paddle.set_device('gpu')


mfcc_extractor = paddleaudio.features.MFCC(
    **mfcc_conf, f_min=0.0, dtype=waveform_tensor.dtype)


def mfcc():
    return mfcc_extractor(waveform_tensor).squeeze(0)


def test_mfcc_cpu(benchmark):
    enable_cpu_device()
    feature_paddleaudio = benchmark(mfcc)
    feature_librosa = librosa.feature.mfcc(waveform, **mel_conf)
    np.testing.assert_array_almost_equal(
        feature_librosa, feature_paddleaudio, decimal=3)


def test_mfcc_gpu(benchmark):
    enable_gpu_device()
    feature_paddleaudio = benchmark(mfcc)
    feature_librosa = librosa.feature.mfcc(waveform, **mel_conf)
    np.testing.assert_array_almost_equal(
        feature_librosa, feature_paddleaudio, decimal=3)


del mel_conf_torchaudio['sample_rate']
mfcc_extractor_torchaudio = torchaudio.transforms.MFCC(
    **mfcc_conf_torchaudio, melkwargs=mel_conf_torchaudio)


def mfcc_torchaudio():
    return mfcc_extractor_torchaudio(waveform_tensor_torch).squeeze(0)


def test_mfcc_cpu_torchaudio(benchmark):
    global waveform_tensor_torch, mfcc_extractor_torchaudio

    mel_extractor_torchaudio = mfcc_extractor_torchaudio.to('cpu')
    waveform_tensor_torch = waveform_tensor_torch.to('cpu')

    feature_paddleaudio = benchmark(mfcc_torchaudio)
    feature_librosa = librosa.feature.mfcc(waveform, **mel_conf)
    np.testing.assert_array_almost_equal(
        feature_librosa, feature_paddleaudio, decimal=3)


def test_mfcc_gpu_torchaudio(benchmark):
    global waveform_tensor_torch, mfcc_extractor_torchaudio

    mel_extractor_torchaudio = mfcc_extractor_torchaudio.to('cuda')
    waveform_tensor_torch = waveform_tensor_torch.to('cuda')

    feature_torchaudio = benchmark(mfcc_torchaudio)
    feature_librosa = librosa.feature.mfcc(waveform, **mel_conf)
    np.testing.assert_array_almost_equal(
        feature_librosa, feature_torchaudio.cpu(), decimal=3)
