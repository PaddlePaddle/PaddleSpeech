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

import numpy as np
import paddle

from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.ssml.xml_processor import MixTextProcessor
from paddlespeech.t2s.frontend.zh_frontend import Frontend


class MixFrontend():
    def __init__(self,
                 g2p_model="pypinyin",
                 phone_vocab_path=None,
                 tone_vocab_path=None):

        self.zh_frontend = Frontend(
            phone_vocab_path=phone_vocab_path, tone_vocab_path=tone_vocab_path)
        self.en_frontend = English(phone_vocab_path=phone_vocab_path)
        self.sp_id = self.zh_frontend.vocab_phones["sp"]
        self.sp_id_numpy = np.array([self.sp_id])
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

    def is_other(self, char):
        if not (self.is_chinese(char) or self.is_alphabet(char)):
            return True
        else:
            return False

    def get_segment(self, text: str) -> List[str]:
        # sentence --> [ch_part, en_part, ch_part, ...]
        segments = []
        types = []
        flag = 0
        temp_seg = ""
        temp_lang = ""

        # Determine the type of each character. type: blank, chinese, alphabet, number, unk and point.
        for ch in text:
            if self.is_chinese(ch):
                types.append("zh")
            elif self.is_alphabet(ch):
                types.append("en")
            else:
                types.append("other")

        assert len(types) == len(text)

        for i in range(len(types)):
            # find the first char of the seg
            if flag == 0:
                temp_seg += text[i]
                temp_lang = types[i]
                flag = 1

            else:
                if temp_lang == "other":
                    if types[i] == temp_lang:
                        temp_seg += text[i]
                    else:
                        temp_seg += text[i]
                        temp_lang = types[i]

                else:
                    if types[i] == temp_lang:
                        temp_seg += text[i]
                    elif types[i] == "other":
                        temp_seg += text[i]
                    else:
                        segments.append((temp_seg, temp_lang))
                        temp_seg = text[i]
                        temp_lang = types[i]
                        flag = 1

        segments.append((temp_seg, temp_lang))

        return segments

    def get_input_ids(self,
                      sentence: str,
                      merge_sentences: bool=False,
                      get_tone_ids: bool=False,
                      add_sp: bool=True,
                      to_tensor: bool=True) -> Dict[str, List[paddle.Tensor]]:
        ''' 1. 添加SSML支持，先列出 文字 和 <say-as>标签内容，
                然后添加到tmpSegments数组里
        '''
        d_inputs = MixTextProcessor.get_dom_split(sentence)
        tmpSegments = []
        for instr in d_inputs:
            ''' 暂时只支持 say-as '''
            if instr.lower().startswith("<say-as"):
                tmpSegments.append((instr, "zh"))
            else:
                tmpSegments.extend(self.get_segment(instr))
        ''' 2. 把zh的merge到一起，避免合成结果中间停顿
        '''
        segments = []
        currentSeg = ["", ""]
        for seg in tmpSegments:
            if seg[1] == "en" or seg[1] == "other":
                if currentSeg[0] == '':
                    segments.append(seg)
                else:
                    currentSeg[0] = "<speak>" + currentSeg[0] + "</speak>"
                    segments.append(tuple(currentSeg))
                    segments.append(seg)
                    currentSeg = ["", ""]
            else:
                if currentSeg[0] == '':
                    currentSeg[0] = seg[0]
                    currentSeg[1] = seg[1]
                else:
                    currentSeg[0] = currentSeg[0] + seg[0]
        if currentSeg[0] != '':
            currentSeg[0] = "<speak>" + currentSeg[0] + "</speak>"
            segments.append(tuple(currentSeg))

        phones_list = []
        result = {}

        for seg in segments:
            content = seg[0]
            lang = seg[1]
            if content != '':
                if lang == "en":
                    input_ids = self.en_frontend.get_input_ids(
                        content, merge_sentences=False, to_tensor=to_tensor)
                else:
                    ''' 3. 把带speak tag的中文和普通文字分开处理
                    '''
                    if content.strip() != "" and \
                        re.match(r".*?<speak>.*?</speak>.*", content, re.DOTALL):
                        input_ids = self.zh_frontend.get_input_ids_ssml(
                            content,
                            merge_sentences=False,
                            get_tone_ids=get_tone_ids,
                            to_tensor=to_tensor)
                    else:
                        input_ids = self.zh_frontend.get_input_ids(
                            content,
                            merge_sentences=False,
                            get_tone_ids=get_tone_ids,
                            to_tensor=to_tensor)
                if add_sp:
                    if to_tensor:
                        input_ids["phone_ids"][-1] = paddle.concat(
                            [input_ids["phone_ids"][-1], self.sp_id_tensor])
                    else:
                        input_ids["phone_ids"][-1] = np.concatenate(
                            (input_ids["phone_ids"][-1], self.sp_id_numpy))

                for phones in input_ids["phone_ids"]:
                    phones_list.append(phones)

        if merge_sentences:
            merge_list = paddle.concat(phones_list)
            # rm the last 'sp' to avoid the noise at the end
            # cause in the training data, no 'sp' in the end
            if (to_tensor and merge_list[-1] == self.sp_id_tensor) or (
                    not to_tensor and merge_list[-1] == self.sp_id_numpy):
                merge_list = merge_list[:-1]
            phones_list = []
            phones_list.append(merge_list)

        result["phone_ids"] = phones_list

        return result
