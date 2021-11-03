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
import os
import random

import numpy as np
import paddle
from paddle.io import Dataset
from paddlenlp.transformers import BertTokenizer
# from speechtask.punctuation_restoration.utils.punct_prepro import load_dataset

__all__ = ["PuncDataset", "PuncDatasetFromBertTokenizer"]


class PuncDataset(Dataset):
    """Representing a Dataset
    superclass
    ----------
    data.Dataset :
        Dataset is a abstract class, representing the real data.
    """

    def __init__(self, train_path, vocab_path, punc_path, seq_len=100):
        # 检查文件是否存在
        print(train_path)
        print(vocab_path)
        assert os.path.exists(train_path), "train文件不存在"
        assert os.path.exists(vocab_path), "词典文件不存在"
        assert os.path.exists(punc_path), "标点文件不存在"
        self.seq_len = seq_len

        self.word2id = self.load_vocab(
            vocab_path, extra_word_list=['<UNK>', '<END>'])
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.punc2id = self.load_vocab(punc_path, extra_word_list=[" "])
        self.id2punc = {k: v for (v, k) in self.punc2id.items()}

        tmp_seqs = open(train_path, encoding='utf-8').readlines()
        self.txt_seqs = [i for seq in tmp_seqs for i in seq.split()]
        # print(self.txt_seqs[:10])
        # with open('./txt_seq', 'w', encoding='utf-8') as w:
        #     print(self.txt_seqs, file=w)
        self.preprocess(self.txt_seqs)
        print('---punc-')
        print(self.punc2id)

    def __len__(self):
        """return the sentence nums in .txt
        """
        return self.in_len

    def __getitem__(self, index):
        """返回指定索引的张量对 (输入文本id的序列 , 其对应的标点id序列)
        Parameters
        ----------
        index : int 索引
        """
        return self.input_data[index], self.label[index]

    def load_vocab(self, vocab_path, extra_word_list=[], encoding='utf-8'):
        n = len(extra_word_list)
        with open(vocab_path, encoding='utf-8') as vf:
            vocab = {word.strip(): i + n for i, word in enumerate(vf)}
        for i, word in enumerate(extra_word_list):
            vocab[word] = i
        return vocab

    def preprocess(self, txt_seqs: list):
        """将文本转为单词和应预测标点的id pair
        Parameters
        ----------
        txt : 文本
            文本每个单词跟随一个空格，符号也跟一个空格
        """
        input_data = []
        label = []
        input_r = []
        label_r = []
        # txt_seqs is a list like: ['char', 'char', 'char', '*，*', 'char', ......]
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
                # print('标点{}：'.format(count), self.punc2id[" "])
                label.append(self.punc2id[" "])
                input_data.append(
                    self.word2id.get(token, self.word2id["<UNK>"]))
                input_r.append(token)
                label_r.append(' ')
            else:
                # print('标点{}：'.format(count), self.punc2id[punc])
                label.append(self.punc2id[punc])
                input_data.append(
                    self.word2id.get(token, self.word2id["<UNK>"]))
                input_r.append(token)
                label_r.append(punc)
        if len(input_data) != len(label):
            assert 'error: length input_data != label'
        # code below is for using 100 as a hidden size
        print(len(input_data))
        self.in_len = len(input_data) // self.seq_len
        len_tmp = self.in_len * self.seq_len
        input_data = input_data[:len_tmp]
        label = label[:len_tmp]

        self.input_data = paddle.to_tensor(
            np.array(input_data, dtype='int64').reshape(-1, self.seq_len))
        self.label = paddle.to_tensor(
            np.array(label, dtype='int64').reshape(-1, self.seq_len))


# unk_token='[UNK]'
# sep_token='[SEP]'
# pad_token='[PAD]'
# cls_token='[CLS]'
# mask_token='[MASK]'


class PuncDatasetFromBertTokenizer(Dataset):
    """Representing a Dataset
    superclass
    ----------
    data.Dataset :
        Dataset is a abstract class, representing the real data.
    """

    def __init__(self,
                 train_path,
                 is_eval,
                 pretrained_token,
                 punc_path,
                 seq_len=100):
        # 检查文件是否存在
        print(train_path)
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_token, do_lower_case=True)
        self.paddingID = self.tokenizer.pad_token_id
        assert os.path.exists(train_path), "train文件不存在"
        assert os.path.exists(punc_path), "标点文件不存在"
        self.seq_len = seq_len

        self.punc2id = self.load_vocab(punc_path, extra_word_list=[" "])
        self.id2punc = {k: v for (v, k) in self.punc2id.items()}

        tmp_seqs = open(train_path, encoding='utf-8').readlines()
        self.txt_seqs = [i for seq in tmp_seqs for i in seq.split()]
        # print(self.txt_seqs[:10])
        # with open('./txt_seq', 'w', encoding='utf-8') as w:
        #     print(self.txt_seqs, file=w)
        if (is_eval):
            self.preprocess(self.txt_seqs)
        else:
            self.preprocess_shift(self.txt_seqs)
        print("data len: %d" % (len(self.input_data)))
        print('---punc-')
        print(self.punc2id)

    def __len__(self):
        """return the sentence nums in .txt
        """
        return self.in_len

    def __getitem__(self, index):
        """返回指定索引的张量对 (输入文本id的序列 , 其对应的标点id序列)
        Parameters
        ----------
        index : int 索引
        """
        return self.input_data[index], self.label[index]

    def load_vocab(self, vocab_path, extra_word_list=[], encoding='utf-8'):
        n = len(extra_word_list)
        with open(vocab_path, encoding='utf-8') as vf:
            vocab = {word.strip(): i + n for i, word in enumerate(vf)}
        for i, word in enumerate(extra_word_list):
            vocab[word] = i
        return vocab

    def preprocess(self, txt_seqs: list):
        """将文本转为单词和应预测标点的id pair
        Parameters
        ----------
        txt : 文本
            文本每个单词跟随一个空格，符号也跟一个空格
        """
        input_data = []
        label = []
        # txt_seqs is a list like: ['char', 'char', 'char', '*，*', 'char', ......]
        count = 0
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
                # print('标点{}：'.format(count), self.punc2id[" "])
                label.append(self.punc2id[" "])
            else:
                label.append(self.punc2id[punc])

        if len(input_data) != len(label):
            assert 'error: length input_data != label'
        # code below is for using 100 as a hidden size

        # print(len(input_data[0]))
        # print(len(label))
        self.in_len = len(input_data) // self.seq_len
        len_tmp = self.in_len * self.seq_len
        input_data = input_data[:len_tmp]
        label = label[:len_tmp]
        # # print(input_data)
        # print(type(input_data))
        # tmp=np.array(input_data)
        # print('--~~~~~~~~~~~~~')
        # print(type(tmp))
        # print(tmp.shape)
        self.input_data = paddle.to_tensor(
            np.array(input_data, dtype='int64').reshape(
                -1, self.seq_len))  #, dtype='int64'
        self.label = paddle.to_tensor(
            np.array(label, dtype='int64').reshape(
                -1, self.seq_len))  #, dtype='int64'

    def preprocess_shift(self, txt_seqs: list):
        """将文本转为单词和应预测标点的id pair
        Parameters
        ----------
        txt : 文本
            文本每个单词跟随一个空格，符号也跟一个空格
        """
        input_data = []
        label = []
        # txt_seqs is a list like: ['char', 'char', 'char', '*，*', 'char', ......]
        count = 0
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
                # print('标点{}：'.format(count), self.punc2id[" "])
                label.append(self.punc2id[" "])
            else:
                label.append(self.punc2id[punc])

        if len(input_data) != len(label):
            assert 'error: length input_data != label'

        # print(len(input_data[0]))
        # print(len(label))
        start = 0
        processed_data = []
        processed_label = []
        while (start < len(input_data) - self.seq_len):
            # end=start+self.seq_len
            end = random.randint(start + self.seq_len // 2,
                                 start + self.seq_len)
            processed_data.append(input_data[start:end])
            processed_label.append(label[start:end])

            start = start + random.randint(1, self.seq_len // 2)

        self.in_len = len(processed_data)
        # # print(input_data)
        # print(type(input_data))
        # tmp=np.array(input_data)
        # print('--~~~~~~~~~~~~~')
        # print(type(tmp))
        # print(tmp.shape)
        self.input_data = processed_data
        #paddle.to_tensor(np.array(processed_data, dtype='int64'))  #, dtype='int64'
        self.label = processed_label
        #paddle.to_tensor(np.array(processed_label, dtype='int64')) #, dtype='int64'


if __name__ == '__main__':
    dataset = PuncDataset()
