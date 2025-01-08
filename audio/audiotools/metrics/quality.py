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
import os

import numpy as np
import paddle

from ..core import AudioSignal


def visqol(
        estimates: AudioSignal,
        references: AudioSignal,
        mode: str="audio", ):  # pragma: no cover
    """ViSQOL score.

    Parameters
    ----------
    estimates : AudioSignal
        Degraded AudioSignal
    references : AudioSignal
        Reference AudioSignal
    mode : str, optional
        'audio' or 'speech', by default 'audio'

    Returns
    -------
    Tensor[float]
        ViSQOL score (MOS-LQO)
    """
    try:
        from pyvisqol import visqol_lib_py
        from pyvisqol.pb2 import visqol_config_pb2
        from pyvisqol.pb2 import similarity_result_pb2
    except ImportError:
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2
        from visqol.pb2 import similarity_result_pb2

    config = visqol_config_pb2.VisqolConfig()
    if mode == "audio":
        target_sr = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        target_sr = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    config.audio.sample_rate = target_sr
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    estimates = estimates.clone().to_mono().resample(target_sr)
    references = references.clone().to_mono().resample(target_sr)

    visqols = []
    for i in range(estimates.batch_size):
        _visqol = api.Measure(
            references.audio_data[i, 0].detach().cpu().numpy().astype(float),
            estimates.audio_data[i, 0].detach().cpu().numpy().astype(float), )
        visqols.append(_visqol.moslqo)
    return paddle.to_tensor(np.array(visqols))


if __name__ == "__main__":
    signal = AudioSignal(paddle.randn([44100]), 44100)
    print(visqol(signal, signal))
