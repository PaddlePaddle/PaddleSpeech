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
"""
A Simple Chinese Phonology using pinyin symbols. 
The G2P conversion converts pinyin string to symbols. Also it can handle string
in Chinese chracters, but due to the complexity of chinese G2P, we can leave 
text -> pinyin to other part of a TTS system. Other NLP techniques may be used
(e.g. tokenization, tagging, NER...)
"""
import re
from itertools import product

from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.core import DefaultConverter
from pypinyin.core import Pinyin
from pypinyin.core import Style

from parakeet.frontend.phonectic import Phonetics
from parakeet.frontend.vocab import Vocab

_punctuations = ['，', '。', '？', '！']
_initials = [
    'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'zh',
    'ch', 'sh', 'r', 'z', 'c', 's'
]
_finals = [
    'ii', 'iii', 'a', 'o', 'e', 'ea', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang',
    'eng', 'er', 'i', 'ia', 'io', 'ie', 'iai', 'iao', 'iou', 'ian', 'ien',
    'iang', 'ieng', 'u', 'ua', 'uo', 'uai', 'uei', 'uan', 'uen', 'uang', 'ueng',
    'v', 've', 'van', 'ven', 'veng'
]
_ernized_symbol = ['&r']
_phones = _initials + _finals + _ernized_symbol + _punctuations
_tones = ['0', '1', '2', '3', '4', '5']

_toned_finals = [final + tone for final, tone in product(_finals, _tones[1:])]
_toned_phonems = _initials + _toned_finals + _ernized_symbol + _punctuations


class ParakeetConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


class ParakeetPinyin(Phonetics):
    def __init__(self):
        self.vocab_phonemes = Vocab(_phones)
        self.vocab_tones = Vocab(_tones)
        self.pinyin_backend = Pinyin(ParakeetConverter())

    def convert_pypinyin_tone3(self, syllables, add_start_end=False):
        phonemes, tones = _convert_to_parakeet_style_pinyin(syllables)

        if add_start_end:
            start = self.vocab_phonemes.start_symbol
            end = self.vocab_phonemes.end_symbol
            phonemes = [start] + phonemes + [end]

            start = self.vocab_tones.start_symbol
            end = self.vocab_tones.end_symbol
            phonemes = [start] + tones + [end]

        phonemes = [
            item for item in phonemes if item in self.vocab_phonemes.stoi
        ]
        tones = [item for item in tones if item in self.vocab_tones.stoi]
        return phonemes, tones

    def phoneticize(self, sentence, add_start_end=False):
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
        syllables = self.pinyin_backend.lazy_pinyin(
            sentence, style=Style.TONE3, strict=True)
        phonemes, tones = self.convert_pypinyin_tone3(
            syllables, add_start_end=add_start_end)
        return phonemes, tones

    def numericalize(self, phonemes, tones):
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
        phoneme_ids = [self.vocab_phonemes.lookup(item) for item in phonemes]
        tone_ids = [self.vocab_tones.lookup(item) for item in tones]
        return phoneme_ids, tone_ids

    def __call__(self, sentence, add_start_end=False):
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
        phonemes, tones = self.phoneticize(
            sentence, add_start_end=add_start_end)
        phoneme_ids, tone_ids = self.numericalize(phonemes, tones)
        return phoneme_ids, tone_ids

    @property
    def vocab_size(self):
        """ Vocab size.
        """
        # 70 = 62 phones + 4 punctuations + 4 special tokens
        return len(self.vocab_phonemes)

    @property
    def tone_vocab_size(self):
        # 10 = 1 non tone + 5 tone + 4 special tokens
        return len(self.vocab_tones)


class ParakeetPinyinWithTone(Phonetics):
    def __init__(self):
        self.vocab = Vocab(_toned_phonems)
        self.pinyin_backend = Pinyin(ParakeetConverter())

    def convert_pypinyin_tone3(self, syllables, add_start_end=False):
        phonemes = _convert_to_parakeet_style_pinyin_with_tone(syllables)

        if add_start_end:
            start = self.vocab_phonemes.start_symbol
            end = self.vocab_phonemes.end_symbol
            phonemes = [start] + phonemes + [end]

        phonemes = [item for item in phonemes if item in self.vocab.stoi]
        return phonemes

    def phoneticize(self, sentence, add_start_end=False):
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
        syllables = self.pinyin_backend.lazy_pinyin(
            sentence, style=Style.TONE3, strict=True)
        phonemes = self.convert_pypinyin_tone3(
            syllables, add_start_end=add_start_end)
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
        phoneme_ids = [self.vocab.lookup(item) for item in phonemes]
        return phoneme_ids

    def __call__(self, sentence, add_start_end=False):
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
        phonemes = self.phoneticize(sentence, add_start_end=add_start_end)
        phoneme_ids = self.numericalize(phonemes)
        return phoneme_ids

    @property
    def vocab_size(self):
        """ Vocab size.
        """
        # 230 = 222 phones + 4 punctuations + 4 special tokens
        return len(self.vocab)


def _convert_to_parakeet_convension(syllable):
    # from pypinyin.Style.TONE3 to parakeet convension
    tone = syllable[-1]
    syllable = syllable[:-1]

    # expansion of o -> uo
    syllable = re.sub(r"([bpmf])o$", r"\1uo", syllable)

    # expansion for iong, ong
    syllable = syllable.replace("iong", "veng").replace("ong", "ueng")

    # expansion for ing, in
    syllable = syllable.replace("ing", "ieng").replace("in", "ien")

    # expansion for un, ui, iu
    syllable = syllable.replace("un", "uen") \
        .replace("ui", "uei") \
        .replace("iu", "iou")

    # rule for variants of i
    syllable = syllable.replace("zi", "zii") \
        .replace("ci", "cii") \
        .replace("si", "sii") \
        .replace("zhi", "zhiii") \
        .replace("chi", "chiii") \
        .replace("shi", "shiii") \
        .replace("ri", "riii")

    # rule for y preceding i, u
    syllable = syllable.replace("yi", "i").replace("yu", "v").replace("y", "i")

    # rule for w
    syllable = syllable.replace("wu", "u").replace("w", "u")

    # rule for v following j, q, x
    syllable = syllable.replace("ju", "jv") \
        .replace("qu", "qv") \
        .replace("xu", "xv")

    return syllable + tone


def _split_syllable(syllable: str):
    global _punctuations

    if syllable in _punctuations:
        # syllables, tones
        return [syllable], ['0']

    syllable = _convert_to_parakeet_convension(syllable)

    tone = syllable[-1]
    syllable = syllable[:-1]

    phones = []
    tones = []

    global _initials
    if syllable[:2] in _initials:
        phones.append(syllable[:2])
        tones.append('0')
        phones.append(syllable[2:])
        tones.append(tone)
    elif syllable[0] in _initials:
        phones.append(syllable[0])
        tones.append('0')
        phones.append(syllable[1:])
        tones.append(tone)
    else:
        phones.append(syllable)
        tones.append(tone)
    return phones, tones


def _convert_to_parakeet_style_pinyin(syllables):
    phones, tones = [], []
    for syllable in syllables:
        p, t = _split_syllable(syllable)
        phones.extend(p)
        tones.extend(t)
    return phones, tones


def _split_syllable_with_tone(syllable: str):
    global _punctuations

    if syllable in _punctuations:
        # syllables
        return [syllable]

    syllable = _convert_to_parakeet_convension(syllable)

    phones = []

    global _initials
    if syllable[:2] in _initials:
        phones.append(syllable[:2])
        phones.append(syllable[2:])
    elif syllable[0] in _initials:
        phones.append(syllable[0])
        phones.append(syllable[1:])
    else:
        phones.append(syllable)
    return phones


def _convert_to_parakeet_style_pinyin_with_tone(syllables):
    phones = []
    for syllable in syllables:
        p = _split_syllable_with_tone(syllable)
        phones.extend(p)
    return phones
