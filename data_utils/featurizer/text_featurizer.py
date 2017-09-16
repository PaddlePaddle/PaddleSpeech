"""Contains the text featurizer class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs


class TextFeaturizer(object):
    """Text featurizer, for processing or extracting features from text.

    Currently, it only supports char-level tokenizing and conversion into
    a list of token indices. Note that the token indexing order follows the
    given vocabulary file.

    :param vocab_filepath: Filepath to load vocabulary for token indices
                           conversion.
    :type specgram_type: basestring
    """

    def __init__(self, vocab_filepath):
        self._vocab_dict, self._vocab_list = self._load_vocabulary_from_file(
            vocab_filepath)

    def featurize(self, text):
        """Convert text string to a list of token indices in char-level.Note
        that the token indexing order follows the given vocabulary file.

        :param text: Text to process.
        :type text: basestring
        :return: List of char-level token indices.
        :rtype: list
        """
        tokens = self._char_tokenize(text)
        return [self._vocab_dict[token] for token in tokens]

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
