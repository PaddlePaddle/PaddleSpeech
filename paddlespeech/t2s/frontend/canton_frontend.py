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
from typing import Dict
from typing import List

import numpy as np
import paddle
import ToJyutping

from paddlespeech.t2s.frontend.zh_normalization.text_normlization import TextNormalizer

INITIALS = [
    'aa', 'aai', 'aak', 'aap', 'aat', 'aau', 'ai', 'au', 'ap', 'at', 'ak', 'a',
    'p', 'b', 'e', 'ts', 't', 'dz', 'd', 'kw', 'k', 'gw', 'g', 'f', 'h', 'l',
    'm', 'ng', 'n', 's', 'y', 'w', 'c', 'z', 'j', 'ong', 'on', 'ou', 'oi', 'ok',
    'o', 'uk', 'ung'
]
INITIALS += ['sp', 'spl', 'spn', 'sil']


def get_lines(cantons: List[str]):
    phones = []
    for canton in cantons:
        for consonant in INITIALS:
            if canton.startswith(consonant):
                if canton.startswith("nga"):
                    c, v = canton[:len(consonant)], canton[len(consonant):]
                    phones = phones + [canton[2:]]
                else:
                    c, v = canton[:len(consonant)], canton[len(consonant):]
                    phones = phones + [c, v]
                break
    return phones


class CantonFrontend():
    def __init__(self, phone_vocab_path: str):
        self.text_normalizer = TextNormalizer()
        self.punc = "：，；。？！“”‘’':,;.?!"

        self.vocab_phones = {}
        if phone_vocab_path:
            with open(phone_vocab_path, 'rt', encoding='utf-8') as f:
                phn_id = [line.strip().split() for line in f.readlines()]
            for phn, id in phn_id:
                self.vocab_phones[phn] = int(id)

    # if merge_sentences, merge all sentences into one phone sequence
    def _g2p(self,
             sentences: List[str],
             merge_sentences: bool = True) -> List[List[str]]:
        phones_list = []
        for sentence in sentences:
            phones_str = ToJyutping.get_jyutping_text(sentence)
            phones_split = get_lines(phones_str.split(' '))
            phones_list.append(phones_split)
        return phones_list

    def _p2id(self, phonemes: List[str]) -> np.ndarray:
        # replace unk phone with sp
        phonemes = [
            phn if phn in self.vocab_phones else "sp" for phn in phonemes
        ]
        phone_ids = [self.vocab_phones[item] for item in phonemes]
        return np.array(phone_ids, np.int64)

    def get_phonemes(self,
                     sentence: str,
                     merge_sentences: bool = True,
                     print_info: bool = False) -> List[List[str]]:
        sentences = self.text_normalizer.normalize(sentence)
        phonemes = self._g2p(sentences, merge_sentences=merge_sentences)
        if print_info:
            print("----------------------------")
            print("text norm results:")
            print(sentences)
            print("----------------------------")
            print("g2p results:")
            print(phonemes)
            print("----------------------------")
        return phonemes

    def get_input_ids(self,
                      sentence: str,
                      merge_sentences: bool = True,
                      print_info: bool = False,
                      to_tensor: bool = True) -> Dict[str, List[paddle.Tensor]]:

        phonemes = self.get_phonemes(sentence,
                                     merge_sentences=merge_sentences,
                                     print_info=print_info)
        result = {}
        temp_phone_ids = []

        for phones in phonemes:
            if phones:
                phone_ids = self._p2id(phones)
                # if use paddle.to_tensor() in onnxruntime, the first time will be too low
                if to_tensor:
                    phone_ids = paddle.to_tensor(phone_ids)
                temp_phone_ids.append(phone_ids)
        if temp_phone_ids:
            result["phone_ids"] = temp_phone_ids
        return result
