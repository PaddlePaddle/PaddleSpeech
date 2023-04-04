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
import re
import unicodedata
from builtins import str as unicode

from paddlespeech.t2s.frontend.normalizer.numbers import normalize_numbers


def normalize(sentence):
    """ Normalize English text.
    """
    # preprocessing
    sentence = unicode(sentence)
    sentence = normalize_numbers(sentence)
    sentence = ''.join(char for char in unicodedata.normalize('NFD', sentence)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
    sentence = sentence.lower()
    sentence = re.sub(r"[^ a-z'.,?!\-]", "", sentence)
    sentence = sentence.replace("i.e.", "that is")
    sentence = sentence.replace("e.g.", "for example")
    return sentence
