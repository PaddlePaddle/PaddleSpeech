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
"""Gen Chinese characters to THCHS30-30 phone lexicon using THCHS30-30's lexicon
file1: THCHS-30/data_thchs30/lm_word/lexicon.txt
file2: THCHS-30/resource/dict/lexicon.txt
"""
import argparse
from collections import defaultdict
from pathlib import Path
from typing import List
from typing import Union

# key: (cn, ('ee', 'er4'))，value: count
cn_phones_counter = defaultdict(int)
# key: cn, value: list of (phones, num)
cn_counter = defaultdict(list)
# key: cn, value: list of (phones, probabilities)
cn_counter_p = defaultdict(list)


def is_Chinese(ch):
    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False


def proc_line(line: str):
    line = line.strip()
    if is_Chinese(line[0]):
        line_list = line.split()
        cn_list = line_list[0]
        phone_list = line_list[1:]
        if len(cn_list) == len(phone_list) / 2:
            new_phone_list = [(phone_list[i], phone_list[i + 1])
                              for i in range(0, len(phone_list), 2)]
            assert len(cn_list) == len(new_phone_list)
            for idx, cn in enumerate(cn_list):
                phones = new_phone_list[idx]
                cn_phones_counter[(cn, phones)] += 1


"""
example lines of output
the first column is a Chinese character
the second is the probability of this pronunciation
and the rest are the phones of this pronunciation
一 0.22 ii i1↩
一 0.45 ii i4↩
一 0.32 ii i2↩
一 0.01 ii i5
"""


def gen_lexicon(lexicon_files: List[Union[str, Path]],
                output_path: Union[str, Path]):
    for file_path in lexicon_files:
        with open(file_path, "r") as f1:
            for line in f1:
                proc_line(line)

    for key in cn_phones_counter:
        cn = key[0]
        cn_counter[cn].append((key[1], cn_phones_counter[key]))

    for key in cn_counter:
        phone_count_list = cn_counter[key]
        count_sum = sum([x[1] for x in phone_count_list])
        for item in phone_count_list:
            p = item[1] / count_sum
            p = round(p, 2)
            if p > 0:
                cn_counter_p[key].append((item[0], p))

    with open(output_path, "w") as wf:
        for key in cn_counter_p:
            phone_p_list = cn_counter_p[key]
            for item in phone_p_list:
                phones, p = item
                wf.write(key + " " + str(p) + " " + " ".join(phones) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gen Chinese characters to phone lexicon for THCHS-30 dataset"
    )
    # A line of word_lexicon:
    # 一丁点 ii i4 d ing1 d ian3
    # the first is word, and the rest are the phones of the word, and the len of phones is twice of the word's len
    parser.add_argument(
        "--lexicon-files",
        type=str,
        default="data/dict/lm_word_lexicon_1 data/dict/lm_word_lexicon_2",
        help="lm_word_lexicon files")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/dict/word.lexicon",
        help="path to save output word2phone lexicon")
    args = parser.parse_args()
    lexicon_files = args.lexicon_files.split(" ")
    output_path = Path(args.output_path).expanduser()

    gen_lexicon(lexicon_files, output_path)
