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
import paddle

import paddleaudio

feat_conf = {
    'sr': 16000,
    'n_fft': 512,
    'hop_length': 128,
    'n_mels': 40,
    'f_min': 0.0,
}


@profile
def test_melspect_cpu(input_shape, times):
    paddle.set_device('cpu')
    x = paddle.randn(input_shape)
    feature_extractor = paddleaudio.features.MelSpectrogram(
        **feat_conf, dtype=x.dtype)
    for i in range(times):
        y = feature_extractor(x)


@profile
def test_melspect_gpu(input_shape, times):
    paddle.set_device('gpu')
    x = paddle.randn(input_shape)
    feature_extractor = paddleaudio.features.MelSpectrogram(
        **feat_conf, dtype=x.dtype)
    for i in range(times):
        y = feature_extractor(x)


input_shape = (16, 48000)  # (N, T)
times = 100
test_melspect_cpu(input_shape, times)
test_melspect_gpu(input_shape, times)
