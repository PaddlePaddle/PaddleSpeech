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
from pathlib import Path

from paddlespeech.t2s.frontend.zh_frontend import Frontend as zhFrontend
from paddlespeech.t2s.utils.error_rate import word_errors

SILENCE_TOKENS = {"sp", "sil", "sp1", "spl"}


def text_cleaner(raw_text):
    text = re.sub('#[1-4]|“|”|（|）', '', raw_text)
    text = text.replace("…。", "。")
    text = re.sub('：|；|——|……|、|…|—', '，', text)
    return text


def get_avg_wer(raw_dict, ref_dict, frontend, output_dir):
    edit_distances = []
    ref_lens = []
    wf_g2p = open(output_dir / "text.g2p", "w")
    wf_ref = open(output_dir / "text.ref.clean", "w")
    for utt_id in raw_dict:
        if utt_id not in ref_dict:
            continue
        raw_text = raw_dict[utt_id]
        text = text_cleaner(raw_text)
        g2p_phones = frontend.get_phonemes(text)
        g2p_phones = sum(g2p_phones, [])
        gt_phones = ref_dict[utt_id].split(" ")
        # delete silence tokens in predicted phones and ground truth phones
        g2p_phones = [phn for phn in g2p_phones if phn not in SILENCE_TOKENS]
        gt_phones = [phn for phn in gt_phones if phn not in SILENCE_TOKENS]
        gt_phones = " ".join(gt_phones)
        g2p_phones = " ".join(g2p_phones)
        wf_ref.write(gt_phones + "(baker_" + utt_id + ")" + "\n")
        wf_g2p.write(g2p_phones + "(baker_" + utt_id + ")" + "\n")
        edit_distance, ref_len = word_errors(gt_phones, g2p_phones)
        edit_distances.append(edit_distance)
        ref_lens.append(ref_len)

    return sum(edit_distances) / sum(ref_lens)


def main():
    parser = argparse.ArgumentParser(description="g2p example.")
    parser.add_argument("--input-dir",
                        default="data/g2p",
                        type=str,
                        help="directory to preprocessed test data.")
    parser.add_argument("--output-dir",
                        default="exp/g2p",
                        type=str,
                        help="directory to save g2p results.")

    args = parser.parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    assert input_dir.is_dir()
    raw_dict, ref_dict = dict(), dict()
    raw_path = input_dir / "text"
    ref_path = input_dir / "text.ref"

    with open(raw_path, "r") as rf:
        for line in rf:
            line = line.strip()
            line_list = line.split(" ")
            utt_id, raw_text = line_list[0], " ".join(line_list[1:])
            raw_dict[utt_id] = raw_text
    with open(ref_path, "r") as rf:
        for line in rf:
            line = line.strip()
            line_list = line.split(" ")
            utt_id, phones = line_list[0], " ".join(line_list[1:])
            ref_dict[utt_id] = phones
    frontend = zhFrontend()
    avg_wer = get_avg_wer(raw_dict, ref_dict, frontend, output_dir)
    print("The avg WER of g2p is:", avg_wer)


if __name__ == "__main__":
    main()
