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

import sentencepiece as spm

from deepspeech.frontend.utility import UNK
from deepspeech.frontend.utility import EOS


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
        self.unit_type = unit_type
        self.unk = UNK
        if vocab_filepath:
            self._vocab_dict, self._id2token, self._vocab_list = self._load_vocabulary_from_file(
                vocab_filepath)
            self.unk_id = self._vocab_list.index(self.unk)
            self.eos_id = self._vocab_list.index(EOS)

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

    def detokenize(self, tokens):
        if self.unit_type == 'char':
            text = self.char_detokenize(tokens)
        elif self.unit_type == 'word':
            text = self.word_detokenize(tokens)
        else:  # spm
            text = self.spm_detokenize(tokens)
        return text

    def featurize(self, text):
        """Convert text string to a list of token indices.

        Args:
            text (str): Text to process.
        
        Returns:
            List[int]: List of token indices.
        """
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            token = token if token in self._vocab_dict else self.unk
            ids.append(self._vocab_dict[token])
        return ids

    def defeaturize(self, idxs):
        """Convert a list of token indices to text string,
        ignore index after eos_id. 

        Args:
            idxs (List[int]): List of token indices.

        Returns:
            str: Text to process.
        """
        tokens = []
        for idx in idxs:
            if idx == self.eos_id:
                break
            tokens.append(self._id2token[idx])
        text = self.detokenize(tokens)
        return text

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

        Returns:
            List[str]: tokens.
        """
        return self._vocab_list

    @property
    def vocab_dict(self):
        """Return the vocabulary in dict.

        Returns:
            Dict[str, int]: token str -> int
        """
        return self._vocab_dict

    def char_tokenize(self, text):
        """Character tokenizer.

        Args:
            text (str): text string.

        Returns:
            List[str]: tokens.
        """
        return list(text.strip())

    def char_detokenize(self, tokens):
        """Character detokenizer.

        Args:
            tokens (List[str]): tokens.

        Returns:
           str: text string.
        """
        return "".join(tokens)

    def word_tokenize(self, text):
        """Word tokenizer, separate by <space>."""
        return text.strip().split()

    def word_detokenize(self, tokens):
        """Word detokenizer, separate by <space>."""
        return " ".join(tokens)

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

    def spm_detokenize(self, tokens, input_format='piece'):
        """spm detokenize.

        Args:
            ids (List[str]): tokens.

        Returns:
            str: text
        """
        if input_format == "piece":

            def decode(l):
                return "".join(self.sp.DecodePieces(l))
        elif input_format == "id":

            def decode(l):
                return "".join(self.sp.DecodeIds(l))

        return decode(tokens)

    def _load_vocabulary_from_file(self, vocab_filepath):
        """Load vocabulary from file."""
        vocab_lines = []
        with open(vocab_filepath, 'r', encoding='utf-8') as file:
            vocab_lines.extend(file.readlines())
        vocab_list = [line[:-1] for line in vocab_lines]
        id2token = dict(
            [(idx, token) for (idx, token) in enumerate(vocab_list)])
        token2id = dict(
            [(token, idx) for (idx, token) in enumerate(vocab_list)])
        return token2id, id2token, vocab_list
