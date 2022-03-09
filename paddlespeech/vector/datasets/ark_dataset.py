# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import random
import numpy as np
import kaldi_python_io as k_io
from paddle.io import Dataset
from paddlespeech.vector.utils.data_utils import batch_pad_right
import paddlespeech.vector.utils as utils
from paddlespeech.vector.utils.utils import read_map_file
from paddlespeech.vector import _logger as log

def ark_collate_fn(batch):
    """
    Custom collate function] for kaldi feats dataset

    Args:
        min_chunk_size: min chunk size of a utterance
        max_chunk_size: max chunk size of a utterance

    Returns:
        ark_collate_fn: collate funtion for dataloader
    """

    data = []
    target = []
    for items in batch:
        for x, y in zip(items[0], items[1]):
            data.append(np.array(x))
            target.append(y)

    data, lengths = batch_pad_right(data)
    return np.array(data, dtype=np.float32), \
           np.array(lengths, dtype=np.float32), \
           np.array(target, dtype=np.long).reshape((len(target), 1))


class KaldiArkDataset(Dataset):
    """
    Dataset used to load kaldi ark/scp files.
    """
    def __init__(self, scp_file, label2utt, min_item_size=1,
                 max_item_size=1, repeat=50, min_chunk_size=200,
                 max_chunk_size=400, select_by_speaker=True):
        self.scp_file = scp_file
        self.scp_reader = None
        self.repeat = repeat
        self.min_item_size = min_item_size
        self.max_item_size = max_item_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._collate_fn = ark_collate_fn
        self._is_select_by_speaker = select_by_speaker
        if utils.is_exist(self.scp_file):
            self.scp_reader = k_io.ScriptReader(self.scp_file)

        label2utts, utt2label = read_map_file(label2utt, key_func=int)
        self.utt_info = list(label2utts.items()) if self._is_select_by_speaker else list(utt2label.items())

    @property
    def collate_fn(self):
        """
        Return a collate funtion.
        """
        return self._collate_fn

    def _random_chunk(self, length):
        chunk_size = random.randint(self.min_chunk_size, self.max_chunk_size)
        if chunk_size >= length:
            return 0, length
        start = random.randint(0, length - chunk_size)
        end = start + chunk_size

        return start, end

    def _select_by_speaker(self, index):
        if self.scp_reader is None or not self.utt_info:
            return []
        index = index % (len(self.utt_info))
        inputs = []
        labels = []
        item_size = random.randint(self.min_item_size, self.max_item_size)
        for loop_idx in range(item_size):
            try:
                utt_index = random.randint(0, len(self.utt_info[index][1])) \
                        % len(self.utt_info[index][1])
                key = self.utt_info[index][1][utt_index]
            except:
                print(index, utt_index, len(self.utt_info[index][1]))
                sys.exit(-1)
            x = self.scp_reader[key]
            x = np.transpose(x)
            bg, end = self._random_chunk(x.shape[-1])
            inputs.append(x[:, bg: end])
            labels.append(self.utt_info[index][0])
        return inputs, labels

    def _select_by_utt(self, index):
        if self.scp_reader is None or len(self.utt_info) == 0:
            return {}
        index = index % (len(self.utt_info))
        key = self.utt_info[index][0]
        x = self.scp_reader[key]
        x = np.transpose(x)
        bg, end = self._random_chunk(x.shape[-1])

        y = self.utt_info[index][1]

        return [x[:, bg: end]], [y]

    def __getitem__(self, index):
        if self._is_select_by_speaker:
            return self._select_by_speaker(index)
        else:
            return self._select_by_utt(index)

    def __len__(self):
        return len(self.utt_info) * self.repeat

    def __iter__(self):
        self._start = 0
        return self

    def __next__(self):
        if self._start < len(self):
            ret = self[self._start]
            self._start += 1
            return ret
        else:
            raise StopIteration
