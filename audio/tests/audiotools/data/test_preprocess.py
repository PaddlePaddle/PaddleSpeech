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
