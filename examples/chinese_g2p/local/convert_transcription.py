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
import argparse
import re

import jieba
from pypinyin import lazy_pinyin
from pypinyin import Style


def extract_pinyin(source, target, use_jieba=False):
    with open(source, 'rt', encoding='utf-8') as fin:
        with open(target, 'wt', encoding='utf-8') as fout:
            for i, line in enumerate(fin):
                if i % 2 == 0:
                    sentence_id, raw_text = line.strip().split()
                    raw_text = re.sub(r'#\d', '', raw_text)
                    if use_jieba:
                        raw_text = jieba.lcut(raw_text)
                    syllables = lazy_pinyin(
                        raw_text,
                        errors='ignore',
                        style=Style.TONE3,
                        neutral_tone_with_five=True)
                    transcription = ' '.join(syllables)
                    fout.write(f'{sentence_id} {transcription}\n')
                else:
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract baker pinyin labels")
    parser.add_argument(
        "input", type=str, help="source file of baker's prosody label file")
    parser.add_argument(
        "output", type=str, help="target file to write pinyin lables")
    parser.add_argument(
        "--use-jieba",
        action='store_true',
        help="use jieba for word segmentation.")
    args = parser.parse_args()
    extract_pinyin(args.input, args.output, use_jieba=args.use_jieba)
