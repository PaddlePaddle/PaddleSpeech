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
"""Recorganize THCHS-30 for MFA
read manifest.train from root-dir
Link *.wav to output-dir
dump *.lab from manifest.train, such as: text、syllable and phone
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
"""
import argparse
import os
from pathlib import Path
from typing import Union


def link_wav(root_dir: Union[str, Path], output_dir: Union[str, Path]):
    wav_scp_path = root_dir / 'wav.scp'
    with open(wav_scp_path, 'r') as rf:
        for line in rf:
            utt, feat = line.strip().split()
            wav_path = feat
            wav_name = wav_path.split("/")[-1]
            new_wav_path = output_dir / wav_name
            os.symlink(wav_path, new_wav_path)


def write_lab(root_dir: Union[str, Path],
              output_dir: Union[str, Path],
              script_type='phone'):
    # script_type can in {'word', 'syllable', 'phone'}
    json_name = 'text.' + script_type
    json_path = root_dir / json_name
    with open(json_path, 'r') as rf:
        for line in rf:
            line = line.strip().split()
            utt_id = line[0]
            context = ' '.join(line[1:])
            transcript_name = utt_id + '.lab'
            transcript_path = output_dir / transcript_name
            with open(transcript_path, 'wt') as wf:
                if script_type == 'word':
                    # add space between chinese char
                    context = ''.join([f + ' ' for f in context])[:-1]
                wf.write(context + "\n")


def reorganize_thchs30(root_dir: Union[str, Path],
                       output_dir: Union[str, Path] = None,
                       script_type='phone'):
    output_dir.mkdir(parents=True, exist_ok=True)
    link_wav(root_dir, output_dir)
    write_lab(root_dir, output_dir, script_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize THCHS-30 dataset for MFA")
    parser.add_argument("--root-dir", type=str, help="path to thchs30 dataset.")
    parser.add_argument("--output-dir",
                        type=str,
                        help="path to save outputs (audio and transcriptions)")

    parser.add_argument("--script-type",
                        type=str,
                        default="phone",
                        help="type of lab ('word'/'syllable'/'phone')")

    args = parser.parse_args()
    root_dir = Path(args.root_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    reorganize_thchs30(root_dir, output_dir, args.script_type)
