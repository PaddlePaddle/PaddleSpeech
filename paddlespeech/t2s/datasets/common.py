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
from typing import List

import librosa
import numpy as np
from paddle.io import Dataset

__all__ = ["AudioSegmentDataset", "AudioDataset", "AudioFolderDataset"]


class AudioSegmentDataset(Dataset):
    """A simple dataset adaptor for audio files to train vocoders.
    Read -> trim silence -> normalize -> extract a segment
    """

    def __init__(self,
                 file_paths: List[Path],
                 sample_rate: int,
                 length: int,
                 top_db: float):
        self.file_paths = file_paths
        self.sr = sample_rate
        self.top_db = top_db
        self.length = length  # samples in the clip

    def __getitem__(self, i):
        fpath = self.file_paths[i]
        y, sr = librosa.load(fpath, self.sr)
        y, _ = librosa.effects.trim(y, top_db=self.top_db)
        y = librosa.util.normalize(y)
        y = y.astype(np.float32)

        # pad or trim
        if y.size <= self.length:
            y = np.pad(y, [0, self.length - len(y)], mode='constant')
        else:
            start = np.random.randint(0, 1 + len(y) - self.length)
            y = y[start:start + self.length]
        return y

    def __len__(self):
        return len(self.file_paths)


class AudioDataset(Dataset):
    """A simple dataset adaptor for the audio files.
    Read -> trim silence -> normalize
    """

    def __init__(self,
                 file_paths: List[Path],
                 sample_rate: int,
                 top_db: float=60):
        self.file_paths = file_paths
        self.sr = sample_rate
        self.top_db = top_db

    def __getitem__(self, i):
        fpath = self.file_paths[i]
        y, sr = librosa.load(fpath, self.sr)
        y, _ = librosa.effects.trim(y, top_db=self.top_db)
        y = librosa.util.normalize(y)
        y = y.astype(np.float32)
        return y

    def __len__(self):
        return len(self.file_paths)


class AudioFolderDataset(AudioDataset):
    def __init__(
            self,
            root,
            sample_rate,
            top_db=60,
            extension=".wav", ):
        root = Path(root).expanduser()
        file_paths = sorted(list(root.rglob("*{}".format(extension))))
        super().__init__(file_paths, sample_rate, top_db)
