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

from pypinyin import lazy_pinyin
from pypinyin import Style

worddict = "./dict/jieba_part.dict.utf8"
newdict = "./dict/word_phones.dict"


def GenPhones(initials, finals, seperate=True):

    phones = []
    for c, v in zip(initials, finals):
        if re.match(r'i\d', v):
            if c in ['z', 'c', 's']:
                v = re.sub('i', 'ii', v)
            elif c in ['zh', 'ch', 'sh', 'r']:
                v = re.sub('i', 'iii', v)
        if c:
            if seperate is True:
                phones.append(c + '0')
            elif seperate is False:
                phones.append(c)
            else:
                print("Not sure whether phone and tone need to be separated")
        if v:
            phones.append(v)
    return phones


with open(worddict, "r") as f1, open(newdict, "w+") as f2:
    for line in f1.readlines():
        word = line.split(" ")[0]
        initials = lazy_pinyin(word,
                               neutral_tone_with_five=True,
                               style=Style.INITIALS)
        finals = lazy_pinyin(word,
                             neutral_tone_with_five=True,
                             style=Style.FINALS_TONE3)

        phones = GenPhones(initials, finals, True)

        temp = " ".join(phones)
        f2.write(word + " " + temp + "\n")
