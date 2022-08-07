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
import re
from typing import Dict
from typing import List

import paddle

from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend


class MixFrontend():
    def __init__(self,
                 g2p_model="pypinyin",
                 phone_vocab_path=None,
                 tone_vocab_path=None):

        self.zh_frontend = Frontend(
            phone_vocab_path=phone_vocab_path, tone_vocab_path=tone_vocab_path)
        self.en_frontend = English(phone_vocab_path=phone_vocab_path)
        self.SENTENCE_SPLITOR = re.compile(r'([：、，；。？！,;?!][”’]?)')
        self.sp_id = self.zh_frontend.vocab_phones["sp"]
        self.sp_id_tensor = paddle.to_tensor([self.sp_id])

    def is_chinese(self, char):
        if char >= '\u4e00' and char <= '\u9fa5':
            return True
        else:
            return False

    def is_alphabet(self, char):
        if (char >= '\u0041' and char <= '\u005a') or (char >= '\u0061' and
                                                       char <= '\u007a'):
            return True
        else:
            return False

    def is_number(self, char):
        if char >= '\u0030' and char <= '\u0039':
            return True
        else:
            return False

    def is_other(self, char):
        if not (self.is_chinese(char) or self.is_number(char) or
                self.is_alphabet(char)):
            return True
        else:
            return False

    def _split(self, text: str) -> List[str]:
        text = re.sub(r'[《》【】<=>{}()（）#&@“”^_|…\\]', '', text)
        text = self.SENTENCE_SPLITOR.sub(r'\1\n', text)
        text = text.strip()
        sentences = [sentence.strip() for sentence in re.split(r'\n+', text)]
        return sentences

    def _distinguish(self, text: str) -> List[str]:
        # sentence --> [ch_part, en_part, ch_part, ...]

        segments = []
        types = []

        flag = 0
        temp_seg = ""
        temp_lang = ""

        # Determine the type of each character. type: blank, chinese, alphabet, number, unk.
        for ch in text:
            if self.is_chinese(ch):
                types.append("zh")
            elif self.is_alphabet(ch):
                types.append("en")
            elif ch == " ":
                types.append("blank")
            elif self.is_number(ch):
                types.append("num")
            else:
                types.append("unk")

        assert len(types) == len(text)

        for i in range(len(types)):

            # find the first char of the seg
            if flag == 0:
                if types[i] != "unk" and types[i] != "blank":
                    temp_seg += text[i]
                    temp_lang = types[i]
                    flag = 1

            else:
                if types[i] == temp_lang or types[i] == "num":
                    temp_seg += text[i]

                elif temp_lang == "num" and types[i] != "unk":
                    temp_seg += text[i]
                    if types[i] == "zh" or types[i] == "en":
                        temp_lang = types[i]

                elif temp_lang == "en" and types[i] == "blank":
                    temp_seg += text[i]

                elif types[i] == "unk":
                    pass

                else:
                    segments.append((temp_seg, temp_lang))

                    if types[i] != "unk" and types[i] != "blank":
                        temp_seg = text[i]
                        temp_lang = types[i]
                        flag = 1
                    else:
                        flag = 0
                        temp_seg = ""
                        temp_lang = ""

        segments.append((temp_seg, temp_lang))

        return segments

    def get_input_ids(self,
                      sentence: str,
                      merge_sentences: bool=True,
                      get_tone_ids: bool=False,
                      add_sp: bool=True,
                      to_tensor: bool=True) -> Dict[str, List[paddle.Tensor]]:

        sentences = self._split(sentence)
        phones_list = []
        result = {}

        for text in sentences:
            phones_seg = []
            segments = self._distinguish(text)
            for seg in segments:
                content = seg[0]
                lang = seg[1]
                if lang == "zh":
                    input_ids = self.zh_frontend.get_input_ids(
                        content,
                        merge_sentences=True,
                        get_tone_ids=get_tone_ids,
                        to_tensor=to_tensor)

                elif lang == "en":
                    input_ids = self.en_frontend.get_input_ids(
                        content, merge_sentences=True, to_tensor=to_tensor)

                phones_seg.append(input_ids["phone_ids"][0])
                if add_sp:
                    phones_seg.append(self.sp_id_tensor)

            phones = paddle.concat(phones_seg)
            phones_list.append(phones)

        if merge_sentences:
            merge_list = paddle.concat(phones_list)
            # rm the last 'sp' to avoid the noise at the end
            # cause in the training data, no 'sp' in the end
            if merge_list[-1] == self.sp_id_tensor:
                merge_list = merge_list[:-1]
            phones_list = []
            phones_list.append(merge_list)

        result["phone_ids"] = phones_list

        return result
