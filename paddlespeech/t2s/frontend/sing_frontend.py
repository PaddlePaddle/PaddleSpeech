# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import re
from typing import Dict
from typing import List

import librosa
import numpy as np
import paddle
from pypinyin import lazy_pinyin


class SingFrontend():
    def __init__(self, pinyin_phone_path: str, phone_vocab_path: str):
        """SVS Frontend

        Args:
            pinyin_phone_path (str): pinyin to phone file path, a 'pinyin|phones' (like: ba|b a ) pair per line.
            phone_vocab_path (str): phone to phone id file path, a 'phone phone id' (like: a 4 ) pair per line.
        """
        self.punc = '[：，；。？！“”‘’\':,;.?!]'

        self.pinyin_phones = {'AP': 'AP', 'SP': 'SP'}
        if pinyin_phone_path:
            with open(pinyin_phone_path, 'rt', encoding='utf-8') as f:
                for line in f.readlines():
                    pinyin_phn = [
                        x.strip() for x in line.split('|') if x.strip() != ''
                    ]
                    self.pinyin_phones[pinyin_phn[0]] = pinyin_phn[1]

        self.vocab_phones = {}
        if phone_vocab_path:
            with open(phone_vocab_path, 'rt', encoding='utf-8') as f:
                phn_id = [line.strip().split() for line in f.readlines()]
            for phn, id in phn_id:
                self.vocab_phones[phn] = int(id)

    def get_phones(self, sentence: str) -> List[int]:
        """get phone list

        Args:
            sentence (str): sentence

        Returns:
            List[int]: phones list

        Example:
            sentence = "你好"
            phones = ['n i', 'h ao']
        """
        # remove all punc
        sentence = re.sub(self.punc, "", sentence)

        # Pypinyin can't solve polyphonic words
        sentence = sentence.replace('最长', '最常').replace('长睫毛', '常睫毛') \
            .replace('那么长', '那么常').replace('多长', '多常') \
            .replace('很长', '很常')

        # lyric
        pinyins = lazy_pinyin(sentence, strict=False)
        # replace unk word with SP
        pinyins = [
            pinyin if pinyin in self.pinyin_phones.keys() else "SP"
            for pinyin in pinyins
        ]
        phones = [
            self.pinyin_phones[pinyin.strip()] for pinyin in pinyins
            if pinyin.strip() in self.pinyin_phones
        ]

        return phones

    def get_note_info(self, note_info: str) -> List[str]:
        note_info = [x.strip() for x in note_info.split('|') if x.strip() != '']
        return note_info

    def process(
        self,
        phones: List[int],
        notes: List[str],
        note_durs: List[float],
    ) -> Dict[str, List[paddle.Tensor]]:
        new_phones = []
        new_notes = []
        new_note_durs = []
        is_slurs = []
        assert len(phones) == len(notes) == len(
            note_durs
        ), "Please check the input, text, notes, note_durs should be the same length."
        for i in range(len(phones)):
            phone = phones[i].split()
            note = notes[i].split()
            note_dur = note_durs[i].split()

            for phn in phone:
                new_phones.append(phn)
                new_notes.append(note[0])
                new_note_durs.append(note_dur[0])
                is_slurs.append(0)

            if len(note) > 1:
                for i in range(1, len(note)):
                    new_phones.append(phone[-1])
                    new_notes.append(note[i])
                    new_note_durs.append(note_dur[i])
                    is_slurs.append(1)

        return new_phones, new_notes, new_note_durs, is_slurs

    def get_input_ids(self,
                      svs_input: Dict[str, str],
                      to_tensor: bool = True) -> Dict[str, List[paddle.Tensor]]:
        """convert input to int/float.

        Args:
            svs_input (Dict[str, str]): include keys: if input_type is phones, phones, notes, note_durs and is_slurs are needed.
            if  input_type is word, text, notes, and note_durs sre needed.
            to_tensor (bool, optional): whether to convert to Tensor. Defaults to True.

        Returns:
            Dict[str, List[paddle.Tensor]]: result include phone_ids, note_ids, note_durs, is_slurs.
        """
        result = {}
        input_type = svs_input['input_type']
        if input_type == 'phoneme':
            assert "phones" in svs_input.keys() and "notes" in svs_input.keys() and "note_durs" in svs_input.keys() and "is_slurs" in svs_input.keys(), \
                "When input_type is phoneme, phones, notes, note_durs, is_slurs should be in the svs_input."
            phones = svs_input["phones"].split()
            notes = svs_input["notes"].split()
            note_durs = svs_input["note_durs"].split()
            is_slurs = svs_input["is_slurs"].split()
            assert len(phones) == len(notes) == len(note_durs) == len(
                is_slurs
            ), "Please check the input, phones, notes, note_durs is_slurs should be the same length."
        elif input_type == "word":
            assert "text" in svs_input.keys() and "notes" in svs_input.keys() and "note_durs" in svs_input.keys(), \
                "When input_type is word, text, notes, note_durs, should be in the svs_input."
            phones = self.get_phones(svs_input['text'])
            notes = self.get_note_info(svs_input['notes'])
            note_durs = self.get_note_info(svs_input['note_durs'])
            phones, notes, note_durs, is_slurs = self.process(
                phones=phones, notes=notes, note_durs=note_durs)

        phone_ids = [self.vocab_phones[phn] for phn in phones]
        phone_ids = np.array(phone_ids, np.int64)
        note_ids = [
            librosa.note_to_midi(note.split("/")[0]) if note != 'rest' else 0
            for note in notes
        ]
        note_ids = np.array(note_ids, np.int64)
        note_durs = np.array(note_durs, np.float32)
        is_slurs = np.array(is_slurs, np.int64)

        if to_tensor:
            phone_ids = paddle.to_tensor(phone_ids)
            note_ids = paddle.to_tensor(note_ids)
            note_durs = paddle.to_tensor(note_durs)
            is_slurs = paddle.to_tensor(is_slurs)

        result['phone_ids'] = [phone_ids]
        result['note_ids'] = [note_ids]
        result['note_durs'] = [note_durs]
        result['is_slurs'] = [is_slurs]

        return result
