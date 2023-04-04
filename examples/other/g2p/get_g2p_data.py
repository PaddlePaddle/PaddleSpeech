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
from collections import defaultdict
from pathlib import Path

from praatio import textgrid


def get_baker_data(root_dir):
    alignment_files = sorted(
        list((root_dir / "PhoneLabeling").rglob("*.interval")))
    text_file = root_dir / "ProsodyLabeling/000001-010000.txt"
    text_file = Path(text_file).expanduser()
    # filter out several files that have errors in annotation
    exclude = {'000611', '000662', '002365', '005107'}
    alignment_files = [f for f in alignment_files if f.stem not in exclude]
    data_dict = defaultdict(dict)
    for alignment_fp in alignment_files:
        alignment = textgrid.openTextgrid(alignment_fp,
                                          includeEmptyIntervals=True)
        # only with baker's annotation
        utt_id = alignment.tierNameList[0].split(".")[0]
        intervals = alignment.tierDict[alignment.tierNameList[0]].entryList
        phones = []
        for interval in intervals:
            label = interval.label
            phones.append(label)
        data_dict[utt_id]["phones"] = phones
    for line in open(text_file, "r"):
        if line.startswith("0"):
            utt_id, raw_text = line.strip().split()
            if utt_id in data_dict:
                data_dict[utt_id]['text'] = raw_text
        else:
            pinyin = line.strip().split()
            if utt_id in data_dict:
                data_dict[utt_id]['pinyin'] = pinyin
    return data_dict


def get_g2p_phones(data_dict, frontend):
    for utt_id in data_dict:
        g2p_phones = frontend.get_phonemes(data_dict[utt_id]['text'])
        data_dict[utt_id]["g2p_phones"] = g2p_phones
    return data_dict


def main():
    parser = argparse.ArgumentParser(description="g2p example.")
    parser.add_argument("--root-dir",
                        default=None,
                        type=str,
                        help="directory to baker dataset.")
    parser.add_argument("--output-dir",
                        default="data/g2p",
                        type=str,
                        help="directory to output.")

    args = parser.parse_args()
    root_dir = Path(args.root_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    assert root_dir.is_dir()
    data_dict = get_baker_data(root_dir)
    raw_path = output_dir / "text"
    ref_path = output_dir / "text.ref"
    wf_raw = open(raw_path, "w")
    wf_ref = open(ref_path, "w")
    for utt_id in data_dict:
        wf_raw.write(utt_id + " " + data_dict[utt_id]['text'] + "\n")
        wf_ref.write(utt_id + " " + " ".join(data_dict[utt_id]['phones']) +
                     "\n")


if __name__ == "__main__":
    main()
