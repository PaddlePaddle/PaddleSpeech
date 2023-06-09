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
import os

import yaml


class Polyphonic():
    def __init__(self):
        with open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'polyphonic.yaml'),
                'r',
                encoding='utf-8') as polyphonic_file:
            # 解析yaml
            polyphonic_dict = yaml.load(polyphonic_file, Loader=yaml.FullLoader)
        self.polyphonic_words = polyphonic_dict["polyphonic"]

    def correct_pronunciation(self, word, pinyin):
        # 词汇被词典收录则返回纠正后的读音
        if word in self.polyphonic_words.keys():
            pinyin = self.polyphonic_words[word]
        # 否则返回原读音
        return pinyin
