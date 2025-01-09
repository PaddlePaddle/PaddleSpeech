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
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/data/test_preprocess.py)
import sys
import tempfile
from pathlib import Path

import paddle
sys.path.append("../..")
from audiotools.core.util import find_audio
from audiotools.core.util import read_sources
from audiotools.data import preprocess


def test_create_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        preprocess.create_csv(
            find_audio("././audio/spk", ext=["wav"]), f.name, loudness=True)


def test_create_csv_with_empty_rows():
    audio_files = find_audio("././audio/spk", ext=["wav"])
    audio_files.insert(0, "")
    audio_files.insert(2, "")

    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        preprocess.create_csv(audio_files, f.name, loudness=True)

        audio_files = read_sources([f.name], remove_empty=True)
        assert len(audio_files[0]) == 1
        audio_files = read_sources([f.name], remove_empty=False)
        assert len(audio_files[0]) == 3
