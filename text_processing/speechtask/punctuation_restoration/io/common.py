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
import codecs
import re
import unicodedata

import ujson

PAD = "<PAD>"
UNK = "<UNK>"
NUM = "<NUM>"
END = "</S>"
SPACE = "_SPACE"


def write_json(filename, dataset):
    with codecs.open(filename, mode="w", encoding="utf-8") as f:
        ujson.dump(dataset, f)


def word_convert(word, keep_number=True, lowercase=True):
    if not keep_number:
        if is_digit(word):
            word = NUM
    if lowercase:
        word = word.lower()
    return word


def is_digit(word):
    try:
        float(word)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(word)
        return True
    except (TypeError, ValueError):
        pass
    result = re.compile(r'^[-+]?[0-9]+,[0-9]+$').match(word)
    if result:
        return True
    return False
