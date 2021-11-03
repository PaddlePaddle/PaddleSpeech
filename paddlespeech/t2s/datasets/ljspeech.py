# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from pathlib import Path

from paddle.io import Dataset

__all__ = ["LJSpeechMetaData"]


class LJSpeechMetaData(Dataset):
    def __init__(self, root):
        self.root = Path(root).expanduser()
        wav_dir = self.root / "wavs"
        csv_path = self.root / "metadata.csv"
        records = []
        speaker_name = "ljspeech"
        with open(str(csv_path), 'rt', encoding='utf-8') as f:
            for line in f:
                filename, _, normalized_text = line.strip().split("|")
                filename = str(wav_dir / (filename + ".wav"))
                records.append([filename, normalized_text, speaker_name])
        self.records = records

    def __getitem__(self, i):
        return self.records[i]

    def __len__(self):
        return len(self.records)
