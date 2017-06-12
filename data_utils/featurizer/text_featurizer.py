from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


class TextFeaturizer(object):
    def __init__(self, vocab_filepath):
        self._vocab_dict, self._vocab_list = self._load_vocabulary_from_file(
            vocab_filepath)

    def text2ids(self, text):
        tokens = self._char_tokenize(text)
        return [self._vocab_dict[token] for token in tokens]

    def ids2text(self, ids):
        return ''.join([self._vocab_list[id] for id in ids])

    @property
    def vocab_size(self):
        return len(self._vocab_list)

    @property
    def vocab_list(self):
        return self._vocab_list

    def _char_tokenize(self, text):
        return list(text.strip())

    def _load_vocabulary_from_file(self, vocab_filepath):
        """Load vocabulary from file."""
        vocab_lines = []
        with open(vocab_filepath, 'r') as file:
            vocab_lines.extend(file.readlines())
        vocab_list = [line[:-1] for line in vocab_lines]
        vocab_dict = dict(
            [(token, id) for (id, token) in enumerate(vocab_list)])
        return vocab_dict, vocab_list
