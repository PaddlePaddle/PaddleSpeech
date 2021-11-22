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
import numpy as np
from paddle.io import Dataset

from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.io.utility import pad_list


class TextDataset(Dataset):
    @classmethod
    def from_file(cls, file_path):
        dataset = cls(file_path)
        return dataset

    def __init__(self, file_path):
        self._manifest = []
        with open(file_path) as f:
            for line in f:
                self._manifest.append(line.strip())

    def __len__(self):
        return len(self._manifest)

    def __getitem__(self, idx):
        return self._manifest[idx]


class TextCollatorSpm():
    def __init__(self, unit_type, vocab_filepath, spm_model_prefix):
        assert (vocab_filepath is not None)
        self.text_featurizer = TextFeaturizer(
            unit_type=unit_type,
            vocab_filepath=vocab_filepath,
            spm_model_prefix=spm_model_prefix)
        self.eos_id = self.text_featurizer.eos_id
        self.blank_id = self.text_featurizer.blank_id

    def __call__(self, batch):
        """
        return type  [List, np.array [B, T], np.array [B, T], np.array[B]]
        """
        keys = []
        texts = []
        texts_input = []
        texts_output = []
        text_lens = []

        for idx, item in enumerate(batch):
            key = item.split(" ")[0].strip()
            text = " ".join(item.split(" ")[1:])
            keys.append(key)
            token_ids = self.text_featurizer.featurize(text)
            texts_input.append(
                np.array([self.eos_id] + token_ids).astype(np.int64))
            texts_output.append(
                np.array(token_ids + [self.eos_id]).astype(np.int64))
            text_lens.append(len(token_ids) + 1)

        ys_input_pad = pad_list(texts_input, self.blank_id).astype(np.int64)
        ys_output_pad = pad_list(texts_output, self.blank_id).astype(np.int64)
        y_lens = np.array(text_lens).astype(np.int64)
        return keys, ys_input_pad, ys_output_pad, y_lens
