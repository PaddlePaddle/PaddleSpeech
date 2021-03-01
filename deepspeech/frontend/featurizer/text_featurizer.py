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
import codecs


class TextFeaturizer(object):
    """Text featurizer, for processing or extracting features from text.

    Currently, it only supports char-level tokenizing and conversion into
    a list of token indices. Note that the token indexing order follows the
    given vocabulary file.

    :param vocab_filepath: Filepath to load vocabulary for token indices
                           conversion.
    :type specgram_type: str
    """

    def __init__(self, vocab_filepath):
        self.unk = '<unk>'
        self._vocab_dict, self._vocab_list = self._load_vocabulary_from_file(
            vocab_filepath)

    def featurize(self, text):
        """Convert text string to a list of token indices in char-level.Note
        that the token indexing order follows the given vocabulary file.

        :param text: Text to process.
        :type text: str
        :return: List of char-level token indices.
        :rtype: list
        """
        tokens = self._char_tokenize(text)
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

    def _char_tokenize(self, text):
        """Character tokenizer."""
        return list(text.strip())

    def _load_vocabulary_from_file(self, vocab_filepath):
        """Load vocabulary from file."""
        vocab_lines = []
        with codecs.open(vocab_filepath, 'r', 'utf-8') as file:
            vocab_lines.extend(file.readlines())
        vocab_list = [line[:-1] for line in vocab_lines]
        vocab_dict = dict(
            [(token, id) for (id, token) in enumerate(vocab_list)])
        return vocab_dict, vocab_list
