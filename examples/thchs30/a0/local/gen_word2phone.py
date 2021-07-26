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


def proc_line(line):
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


def gen_lexicon(root_dir: Union[str, Path], output_dir: Union[str, Path]):
    root_dir = Path(root_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    file1 = root_dir / "lm_word_lexicon_1"
    file2 = root_dir / "lm_word_lexicon_2"
    write_file = output_dir / "word.lexicon"

    with open(file1, "r") as f1:
        for line in f1:
            proc_line(line)
    with open(file2, "r") as f2:
        for line in f2:
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
    with open(write_file, "w") as wf:
        for key in cn_counter_p:
            phone_p_list = cn_counter_p[key]
            for item in phone_p_list:
                phones, p = item
                wf.write(key + " " + str(p) + " " + " ".join(phones) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gen Chinese characters to phone lexicon for THCHS-30 dataset"
    )
    parser.add_argument(
        "--root-dir", type=str, help="dir to thchs30 lm_word_lexicons")
    parser.add_argument("--output-dir", type=str, help="path to save outputs")
    args = parser.parse_args()
    gen_lexicon(args.root_dir, args.output_dir)
