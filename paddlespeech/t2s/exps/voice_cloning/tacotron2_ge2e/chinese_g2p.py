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
from typing import List
from typing import Tuple

from pypinyin import lazy_pinyin
from pypinyin import Style

from paddlespeech.t2s.exps.voice_cloning.tacotron2_ge2e.preprocess_transcription import split_syllable


def convert_to_pinyin(text: str) -> List[str]:
    """convert text into list of syllables, other characters that are not chinese, thus
    cannot be converted to pinyin are splited.
    """
    syllables = lazy_pinyin(
        text, style=Style.TONE3, neutral_tone_with_five=True)
    return syllables


def convert_sentence(text: str) -> List[Tuple[str]]:
    """convert a sentence into two list: phones and tones"""
    syllables = convert_to_pinyin(text)
    phones = []
    tones = []
    for syllable in syllables:
        p, t = split_syllable(syllable)
        phones.extend(p)
        tones.extend(t)

    return phones, tones
