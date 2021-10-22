# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from abc import ABC
from abc import abstractmethod

from g2p_en import G2p
from g2pM import G2pM

from parakeet.frontend.vocab import Vocab
from parakeet.frontend.normalizer.normalizer import normalize
from parakeet.frontend.punctuation import get_punctuations

# discard opencc untill we find an easy solution to install it on windows
# from opencc import OpenCC

__all__ = ["Phonetics", "English", "EnglishCharacter", "Chinese"]


class Phonetics(ABC):
    @abstractmethod
    def __call__(self, sentence):
        pass

    @abstractmethod
    def phoneticize(self, sentence):
        pass

    @abstractmethod
    def numericalize(self, phonemes):
        pass


class English(Phonetics):
    """ Normalize the input text sequence and convert into pronunciation id sequence.
    """

    def __init__(self):
        self.backend = G2p()
        self.phonemes = list(self.backend.phonemes)
        self.punctuations = get_punctuations("en")
        self.vocab = Vocab(self.phonemes + self.punctuations)

    def phoneticize(self, sentence):
        """ Normalize the input text sequence and convert it into pronunciation sequence.

        Parameters
        -----------
        sentence: str
            The input text sequence.

        Returns
        ----------
        List[str]
            The list of pronunciation sequence.
        """
        start = self.vocab.start_symbol
        end = self.vocab.end_symbol
        phonemes = ([] if start is None else [start]) \
                   + self.backend(sentence) \
                   + ([] if end is None else [end])
        phonemes = [item for item in phonemes if item in self.vocab.stoi]
        return phonemes

    def numericalize(self, phonemes):
        """ Convert pronunciation sequence into pronunciation id sequence.

        Parameters
        -----------
        phonemes: List[str]
            The list of pronunciation sequence.

        Returns
        ----------
        List[int]
            The list of pronunciation id sequence.
        """
        ids = [
            self.vocab.lookup(item) for item in phonemes
            if item in self.vocab.stoi
        ]
        return ids

    def reverse(self, ids):
        """ Reverse the list of pronunciation id sequence to a list of pronunciation sequence.

        Parameters
        -----------
        ids: List[int]
            The list of pronunciation id sequence.

        Returns
        ----------
        List[str]
            The list of pronunciation sequence.
        """
        return [self.vocab.reverse(i) for i in ids]

    def __call__(self, sentence):
        """ Convert the input text sequence into pronunciation id sequence.

        Parameters
        -----------
        sentence: str
            The input text sequence.

        Returns
        ----------
        List[str]
            The list of pronunciation id sequence.
        """
        return self.numericalize(self.phoneticize(sentence))

    @property
    def vocab_size(self):
        """ Vocab size.
        """
        return len(self.vocab)


class EnglishCharacter(Phonetics):
    """ Normalize the input text sequence and convert it into character id sequence.
    """

    def __init__(self):
        self.backend = G2p()
        self.graphemes = list(self.backend.graphemes)
        self.punctuations = get_punctuations("en")
        self.vocab = Vocab(self.graphemes + self.punctuations)

    def phoneticize(self, sentence):
        """ Normalize the input text sequence.

        Parameters
        -----------
        sentence: str
            The input text sequence.

        Returns
        ----------
        str
            A text sequence after normalize.
        """
        words = normalize(sentence)
        return words

    def numericalize(self, sentence):
        """ Convert a text sequence into ids.

        Parameters
        -----------
        sentence: str
            The input text sequence.

        Returns
        ----------
        List[int]
            List of a character id sequence.
        """
        ids = [
            self.vocab.lookup(item) for item in sentence
            if item in self.vocab.stoi
        ]
        return ids

    def reverse(self, ids):
        """ Convert a character id sequence into text.

        Parameters
        -----------
        ids: List[int]
            List of a character id sequence.

        Returns
        ----------
        str
            The input text sequence.

        """
        return [self.vocab.reverse(i) for i in ids]

    def __call__(self, sentence):
        """ Normalize the input text sequence and convert it into character id sequence.
        Parameters
        -----------
        sentence: str
            The input text sequence.

        Returns
        ----------
        List[int]
            List of a character id sequence.
        """
        return self.numericalize(self.phoneticize(sentence))

    @property
    def vocab_size(self):
        """ Vocab size.
        """
        return len(self.vocab)


class Chinese(Phonetics):
    """Normalize Chinese text sequence and convert it into ids.
    """

    def __init__(self):
        # self.opencc_backend = OpenCC('t2s.json')
        self.backend = G2pM()
        self.phonemes = self._get_all_syllables()
        self.punctuations = get_punctuations("cn")
        self.vocab = Vocab(self.phonemes + self.punctuations)

    def _get_all_syllables(self):
        all_syllables = set([
            syllable for k, v in self.backend.cedict.items() for syllable in v
        ])
        return list(all_syllables)

    def phoneticize(self, sentence):
        """ Normalize the input text sequence and convert it into pronunciation sequence.

        Parameters
        -----------
        sentence: str
            The input text sequence.

        Returns
        ----------
        List[str]
            The list of pronunciation sequence.
        """
        # simplified = self.opencc_backend.convert(sentence)
        simplified = sentence
        phonemes = self.backend(simplified)
        start = self.vocab.start_symbol
        end = self.vocab.end_symbol
        phonemes = ([] if start is None else [start]) \
                   + phonemes \
                   + ([] if end is None else [end])
        return self._filter_symbols(phonemes)

    def _filter_symbols(self, phonemes):
        cleaned_phonemes = []
        for item in phonemes:
            if item in self.vocab.stoi:
                cleaned_phonemes.append(item)
            else:
                for char in item:
                    if char in self.vocab.stoi:
                        cleaned_phonemes.append(char)
        return cleaned_phonemes

    def numericalize(self, phonemes):
        """ Convert pronunciation sequence into pronunciation id sequence.

        Parameters
        -----------
        phonemes: List[str]
            The list of pronunciation sequence.

        Returns
        ----------
        List[int]
            The list of pronunciation id sequence.
        """
        ids = [self.vocab.lookup(item) for item in phonemes]
        return ids

    def __call__(self, sentence):
        """ Convert the input text sequence into pronunciation id sequence.

        Parameters
        -----------
        sentence: str
            The input text sequence.

        Returns
        ----------
        List[str]
            The list of pronunciation id sequence.
        """
        return self.numericalize(self.phoneticize(sentence))

    @property
    def vocab_size(self):
        """ Vocab size.
        """
        return len(self.vocab)

    def reverse(self, ids):
        """ Reverse the list of pronunciation id sequence to a list of pronunciation sequence.

        Parameters
        -----------
        ids: List[int]
            The list of pronunciation id sequence.

        Returns
        ----------
        List[str]
            The list of pronunciation sequence.
        """
        return [self.vocab.reverse(i) for i in ids]
