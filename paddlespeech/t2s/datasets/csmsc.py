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
import os
from pathlib import Path

from paddle.io import Dataset

__all__ = ["CSMSCMetaData"]


class CSMSCMetaData(Dataset):
    def __init__(self, root):
        """
        :param root: the path of baker dataset
        """
        self.root = os.path.abspath(root)
        records = []
        index = 1
        self.meta_info = ["file_path", "text", "pinyin"]

        metadata_path = os.path.join(root, "ProsodyLabeling/000001-010000.txt")
        wav_dirs = os.path.join(self.root, "Wave")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            while True:
                line1 = f.readline().strip()
                if not line1:
                    break
                line2 = f.readline().strip()
                strs = line1.split()
                wav_fname = line1.split()[0].strip() + '.wav'
                wav_filepath = os.path.join(wav_dirs, wav_fname)
                text = strs[1].strip()
                pinyin = line2
                records.append([wav_filepath, text, pinyin])

        self.records = records

    def __getitem__(self, i):
        return self.records[i]

    def __len__(self):
        return len(self.records)

    def get_meta_info(self):
        return self.meta_info
