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

from ..utility import EOS
from ..utility import load_dict
from ..utility import SPACE
from ..utility import UNK

__all__ = ["TextFeaturizer"]


class TextFeaturizer():
    def __init__(self,
                 unit_type,
                 vocab_filepath,
                 spm_model_prefix=None,
                 maskctc=False):
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
        self.maskctc = maskctc

        if vocab_filepath:
            self.vocab_dict, self._id2token, self.vocab_list, self.unk_id, self.eos_id = self._load_vocabulary_from_file(
                vocab_filepath, maskctc)
            self.vocab_size = len(self.vocab_list)

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
            text (str): Text.

        Returns:
            List[int]: List of token indices.
        """
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            token = token if token in self.vocab_dict else self.unk
            ids.append(self.vocab_dict[token])
        return ids

    def defeaturize(self, idxs):
        """Convert a list of token indices to text string,
        ignore index after eos_id.

        Args:
            idxs (List[int]): List of token indices.

        Returns:
            str: Text.
        """
        tokens = []
        for idx in idxs:
            if idx == self.eos_id:
                break
            tokens.append(self._id2token[idx])
        text = self.detokenize(tokens)
        return text

    def char_tokenize(self, text):
        """Character tokenizer.

        Args:
            text (str): text string.

        Returns:
            List[str]: tokens.
        """
        text = text.strip()
        text = text.replace(" ", SPACE)
        return list(text)

    def char_detokenize(self, tokens):
        """Character detokenizer.

        Args:
            tokens (List[str]): tokens.

        Returns:
           str: text string.
        """
        tokens = tokens.replace(SPACE, " ")
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

    def _load_vocabulary_from_file(self, vocab_filepath: str, maskctc: bool):
        """Load vocabulary from file."""
        vocab_list = load_dict(vocab_filepath, maskctc)
        assert vocab_list is not None
        assert SPACE in vocab_list

        id2token = dict(
            [(idx, token) for (idx, token) in enumerate(vocab_list)])
        token2id = dict(
            [(token, idx) for (idx, token) in enumerate(vocab_list)])

        unk_id = vocab_list.index(UNK) if UNK in vocab_list else -1
        eos_id = vocab_list.index(EOS) if EOS in vocab_list else -1

        return token2id, id2token, vocab_list, unk_id, eos_id
