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
import paddle
from paddle.io import Dataset
from paddlenlp.transformers import ErnieTokenizer

__all__ = ["PuncDataset", "PuncDatasetFromErnieTokenizer"]


class PuncDataset(Dataset):
    def __init__(self, train_path, vocab_path, punc_path, seq_len=100):
        self.seq_len = seq_len

        self.word2id = self.load_vocab(vocab_path,
                                       extra_word_list=['<UNK>', '<END>'])
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.punc2id = self.load_vocab(punc_path, extra_word_list=[" "])
        self.id2punc = {k: v for (v, k) in self.punc2id.items()}

        tmp_seqs = open(train_path, encoding='utf-8').readlines()
        self.txt_seqs = [i for seq in tmp_seqs for i in seq.split()]
        self.preprocess(self.txt_seqs)

    def __len__(self):
        """return the sentence nums in .txt
        """
        return self.in_len

    def __getitem__(self, index):
        return self.input_data[index], self.label[index]

    def load_vocab(self, vocab_path, extra_word_list=[], encoding='utf-8'):
        n = len(extra_word_list)
        with open(vocab_path, encoding='utf-8') as vf:
            vocab = {word.strip(): i + n for i, word in enumerate(vf)}
        for i, word in enumerate(extra_word_list):
            vocab[word] = i
        return vocab

    def preprocess(self, txt_seqs: list):
        input_data = []
        label = []
        input_r = []
        label_r = []

        count = 0
        length = len(txt_seqs)
        for token in txt_seqs:
            count += 1
            if count == length:
                break
            if token in self.punc2id:
                continue
            punc = txt_seqs[count]
            if punc not in self.punc2id:
                label.append(self.punc2id[" "])
                input_data.append(self.word2id.get(token,
                                                   self.word2id["<UNK>"]))
                input_r.append(token)
                label_r.append(' ')
            else:
                label.append(self.punc2id[punc])
                input_data.append(self.word2id.get(token,
                                                   self.word2id["<UNK>"]))
                input_r.append(token)
                label_r.append(punc)
        if len(input_data) != len(label):
            assert 'error: length input_data != label'

        self.in_len = len(input_data) // self.seq_len
        len_tmp = self.in_len * self.seq_len
        input_data = input_data[:len_tmp]
        label = label[:len_tmp]

        self.input_data = paddle.to_tensor(
            np.array(input_data, dtype='int64').reshape(-1, self.seq_len))
        self.label = paddle.to_tensor(
            np.array(label, dtype='int64').reshape(-1, self.seq_len))


class PuncDatasetFromErnieTokenizer(Dataset):
    def __init__(self,
                 train_path,
                 punc_path,
                 pretrained_token='ernie-1.0',
                 seq_len=100):
        self.tokenizer = ErnieTokenizer.from_pretrained(pretrained_token)
        self.paddingID = self.tokenizer.pad_token_id
        self.seq_len = seq_len
        self.punc2id = self.load_vocab(punc_path, extra_word_list=[" "])
        self.id2punc = {k: v for (v, k) in self.punc2id.items()}
        tmp_seqs = open(train_path, encoding='utf-8').readlines()
        self.txt_seqs = [i for seq in tmp_seqs for i in seq.split()]
        self.preprocess(self.txt_seqs)

    def __len__(self):
        return self.in_len

    def __getitem__(self, index):
        return self.input_data[index], self.label[index]

    def load_vocab(self, vocab_path, extra_word_list=[], encoding='utf-8'):
        n = len(extra_word_list)
        with open(vocab_path, encoding='utf-8') as vf:
            vocab = {word.strip(): i + n for i, word in enumerate(vf)}
        for i, word in enumerate(extra_word_list):
            vocab[word] = i
        return vocab

    def preprocess(self, txt_seqs: list):
        input_data = []
        label = []
        count = 0
        print("Preprocessing in PuncDatasetFromErnieTokenizer...")
        for i in range(len(txt_seqs) - 1):
            word = txt_seqs[i]
            punc = txt_seqs[i + 1]
            if word in self.punc2id:
                continue

            token = self.tokenizer(word)
            x = token["input_ids"][1:-1]
            input_data.extend(x)

            for i in range(len(x) - 1):
                label.append(self.punc2id[" "])

            if punc not in self.punc2id:
                label.append(self.punc2id[" "])
            else:
                label.append(self.punc2id[punc])

        if len(input_data) != len(label):
            assert 'error: length input_data != label'

        self.in_len = len(input_data) // self.seq_len
        len_tmp = self.in_len * self.seq_len
        input_data = input_data[:len_tmp]
        label = label[:len_tmp]
        self.input_data = np.array(input_data,
                                   dtype='int64').reshape(-1, self.seq_len)
        self.label = np.array(label, dtype='int64').reshape(-1, self.seq_len)
