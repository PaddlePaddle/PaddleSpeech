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
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="text normalization example.")
    parser.add_argument(
        "--test-file",
        default="data/textnorm_test_cases.txt",
        type=str,
        help="path of text normalization test file.")
    parser.add_argument(
        "--output-dir",
        default="data/textnorm",
        type=str,
        help="directory to output.")

    args = parser.parse_args()
    test_file = Path(args.test_file).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "text"
    ref_path = output_dir / "text.ref"
    wf_raw = open(raw_path, "w")
    wf_ref = open(ref_path, "w")

    with open(test_file, "r") as rf:
        for i, line in enumerate(rf):
            raw_text, normed_text = line.strip().split("|")
            wf_raw.write("utt_" + str(i) + " " + raw_text + "\n")
            wf_ref.write("utt_" + str(i) + " " + normed_text + "\n")


if __name__ == "__main__":
    main()
