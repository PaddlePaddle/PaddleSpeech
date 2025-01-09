# MIT License, Copyright (c) 2023-Present, Descript.
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
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/test_post.py)
import sys
from pathlib import Path

sys.path.append("../..")
from audiotools import AudioSignal
from audiotools import post
from audiotools import transforms


def test_audio_table():
    tfm = transforms.LowPass()

    audio_dict = {}

    audio_dict["inputs"] = [
        AudioSignal.excerpt("./audio/spk/f10_script4_produced.wav", duration=5)
        for _ in range(3)
    ]
    audio_dict["outputs"] = []
    for i in range(3):
        x = audio_dict["inputs"][i]

        kwargs = tfm.instantiate()
        output = tfm(x.clone(), **kwargs)
        audio_dict["outputs"].append(output)

    post.audio_table(audio_dict)
