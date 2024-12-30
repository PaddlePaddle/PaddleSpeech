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
from pprint import pformat
from typing import Union

import sentencepiece as spm

from ..utility import BLANK
from ..utility import EOS
from ..utility import load_dict
from ..utility import MASKCTC
from ..utility import SOS
from ..utility import SPACE
from ..utility import UNK
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ["TextFeaturizer"]


class TextFeaturizer():
    def __init__(self, unit_type, vocab, spm_model_prefix=None, maskctc=False):
        """Text featurizer, for processing or extracting features from text.

        Currently, it supports char/word/sentence-piece level tokenizing and conversion into
        a list of token indices. Note that the token indexing order follows the
        given vocabulary file.

        Args:
            unit_type (str): unit type, e.g. char, word, spm
            vocab Option[str, list]: Filepath to load vocabulary for token indices conversion, or vocab list.
            spm_model_prefix (str, optional): spm model prefix. Defaults to None.
        """
        assert unit_type in ('char', 'spm', 'word')
        self.unit_type = unit_type
        self.unk = UNK
        self.maskctc = maskctc
        self.vocab_path_or_list = vocab

        if self.vocab_path_or_list:
            self.vocab_dict, self._id2token, self.vocab_list, self.unk_id, self.eos_id, self.blank_id = self._load_vocabulary_from_file(
                vocab, maskctc)
            self.vocab_size = len(self.vocab_list)
        else:
            logger.warning(
                "TextFeaturizer: not have vocab file or vocab list. Only Tokenizer can use, can not convert to token idx"
            )

        if unit_type == 'spm':
            spm_model = spm_model_prefix + '.model'
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(spm_model)

    def tokenize(self, text, replace_space=True):
        """tokenizer split text into text tokens"""
        if self.unit_type == 'char':
            tokens = self.char_tokenize(text, replace_space)
        elif self.unit_type == 'word':
            tokens = self.word_tokenize(text)
        else:  # spm
            tokens = self.spm_tokenize(text)
        return tokens

    def detokenize(self, tokens):
        """tokenizer convert text tokens back to text"""
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
        assert self.vocab_path_or_list, "toidx need vocab path or vocab list"
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token not in self.vocab_dict:
                logger.debug(f"Text Token: {token} -> {self.unk}")
                token = self.unk
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
        assert self.vocab_path_or_list, "toidx need vocab path or vocab list"
        tokens = []
        # unwrap `idxs`` like `[[1,2,3]]`
        if idxs and isinstance(idxs[0], (list, tuple)) and len(idxs) == 1:
            idxs = idxs[0]

        for idx in idxs:
            if idx == self.eos_id:
                break
            tokens.append(self._id2token[idx])
        text = self.detokenize(tokens)
        return text

    def char_tokenize(self, text, replace_space=True):
        """Character tokenizer.

        Args:
            text (str): text string.
            replace_space (bool): False only used by build_vocab.py.

        Returns:
            List[str]: tokens.
        """
        text = text.strip()
        if replace_space:
            tokens = [SPACE if item == " " else item for item in list(text)]
        else:
            tokens = list(text)
        return tokens

    def char_detokenize(self, tokens):
        """Character detokenizer.

        Args:
            tokens (List[str]): tokens.

        Returns:
           str: text string.
        """
        tokens = [t.replace(SPACE, " ") for t in tokens]
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

    def _load_vocabulary_from_file(self, vocab: Union[str, list],
                                   maskctc: bool):
        """Load vocabulary from file."""
        if isinstance(vocab, list):
            vocab_list = vocab
        else:
            vocab_list = load_dict(vocab, maskctc)
        assert vocab_list is not None
        logger.debug(f"Vocab: {pformat(vocab_list)}")

        id2token = dict(
            [(idx, token) for (idx, token) in enumerate(vocab_list)])
        token2id = dict(
            [(token, idx) for (idx, token) in enumerate(vocab_list)])

        blank_id = vocab_list.index(BLANK) if BLANK in vocab_list else -1
        maskctc_id = vocab_list.index(MASKCTC) if MASKCTC in vocab_list else -1
        unk_id = vocab_list.index(UNK) if UNK in vocab_list else -1
        eos_id = vocab_list.index(EOS) if EOS in vocab_list else -1
        sos_id = vocab_list.index(SOS) if SOS in vocab_list else -1
        space_id = vocab_list.index(SPACE) if SPACE in vocab_list else -1

        logger.debug(f"BLANK id: {blank_id}")
        logger.debug(f"UNK id: {unk_id}")
        logger.debug(f"EOS id: {eos_id}")
        logger.debug(f"SOS id: {sos_id}")
        logger.debug(f"SPACE id: {space_id}")
        logger.debug(f"MASKCTC id: {maskctc_id}")
        return token2id, id2token, vocab_list, unk_id, eos_id, blank_id
