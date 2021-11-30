#!/usr/bin/env python3
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
"""format manifest into wav.scp text.word [text.syllable text.phone]"""
import argparse
from pathlib import Path
from typing import Union

import jsonlines

key_whitelist = set(['feat', 'text', 'syllable', 'phone'])
filename = {
    'text': 'text.word',
    'syllable': 'text.syllable',
    'phone': 'text.phone',
    'feat': 'wav.scp',
}


def dump_manifest(manifest_path, output_dir: Union[str, Path]):

    output_dir = Path(output_dir).expanduser()
    manifest_path = Path(manifest_path).expanduser()

    with jsonlines.open(str(manifest_path), 'r') as reader:
        manifest_jsons = list(reader)

    first_line = manifest_jsons[0]
    file_map = {}

    for k in first_line.keys():
        if k not in key_whitelist:
            continue
        file_map[k] = open(output_dir / filename[k], 'w')

    for line_json in manifest_jsons:
        for k in line_json.keys():
            if k not in key_whitelist:
                continue
            file_map[k].write(line_json['utt'] + ' ' + line_json[k] + '\n')

    for _, file in file_map.items():
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="dump manifest to wav.scp text.word ...")
    parser.add_argument("--manifest-path", type=str, help="path to manifest")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to save outputs(audio and transcriptions)")
    args = parser.parse_args()
    dump_manifest(args.manifest_path, args.output_dir)
