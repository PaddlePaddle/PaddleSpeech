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
"""Contains the text featurizer class."""

import os
import sentencepiece as spm

from deepspeech.frontend.utility import UNK


class TextFeaturizer(object):
    def __init__(self, unit_type, vocab_filepath, spm_model_prefix=None):
        """Text featurizer, for processing or extracting features from text.

        Currently, it supports char/word/sentence-piece level tokenizing and conversion into
        a list of token indices. Note that the token indexing order follows the
        given vocabulary file.

        Args:
            unit_type (str): unit type, e.g. char, word, spm
            vocab_filepath (str): Filepath to load vocabulary for token indices conversion.
            spm_model_prefix (str, optional): spm model prefix. Defaults to None.
        """
        assert unit_type in ('char', 'spm', 'word')
        self.unk = UNK
        self.unit_type = unit_type
        self._vocab_dict, self._vocab_list = self._load_vocabulary_from_file(
            vocab_filepath)

        if unit_type == 'spm':
            spm_model = spm_model_prefix + '.model'
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(spm_model)

    def tokenize(self, text):
        if self.unit_type == 'char':
            tokens = self.char_tokenize(text)
        elif self.unit_type == 'word':
            tokens = self.word_tokenize(text)
        else:  # spm
            tokens = self.spm_tokenize(text)
        return tokens

    def featurize(self, text):
        """Convert text string to a list of token indices in char-level.Note
        that the token indexing order follows the given vocabulary file.

        :param text: Text to process.
        :type text: str
        :return: List of char-level token indices.
        :rtype: List[int]
        """
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            token = token if token in self._vocab_dict else self.unk
            ids.append(self._vocab_dict[token])
        return ids

    @property
    def vocab_size(self):
        """Return the vocabulary size.

        :return: Vocabulary size.
        :rtype: int
        """
        return len(self._vocab_list)

    @property
    def vocab_list(self):
        """Return the vocabulary in list.

        :return: Vocabulary in list.
        :rtype: list
        """
        return self._vocab_list

    def char_tokenize(self, text):
        """Character tokenizer."""
        return list(text.strip())

    def word_tokenize(self, text):
        """Word tokenizer, spearte by <space>."""
        return text.strip().split()

    def spm_tokenize(self, text):
        """spm tokenize.

        Args:
            text (str): text string.

        Returns:
            List[str]: sentence pieces str code
        """
        stats = {"num_empty": 0, "num_filtered": 0}

        def valid(line):
            return True

        def encode(l):
            return self.sp.EncodeAsPieces(l)

        def encode_line(line):
            line = line.strip()
            if len(line) > 0:
                line = encode(line)
                if valid(line):
                    return line
                else:
                    stats["num_filtered"] += 1
            else:
                stats["num_empty"] += 1
            return None

        enc_line = encode_line(text)
        return enc_line

    def _load_vocabulary_from_file(self, vocab_filepath):
        """Load vocabulary from file."""
        vocab_lines = []
        with open(vocab_filepath, 'r', encoding='utf-8') as file:
            vocab_lines.extend(file.readlines())
        vocab_list = [line[:-1] for line in vocab_lines]
        vocab_dict = dict(
            [(token, id) for (id, token) in enumerate(vocab_list)])
        return vocab_dict, vocab_list
