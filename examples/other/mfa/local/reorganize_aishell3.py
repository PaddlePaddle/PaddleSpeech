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
"""Script to reorganize AISHELL-3 dataset so as to use Montreal Force
Aligner to align transcription and audio.

Please refer to https://montreal-forced-aligner.readthedocs.io/en/latest/data_prep.html
for more details about Montreal Force Aligner's requirements on cotpus.

For scripts to reorganize other corpus, please refer to 
 https://github.com/MontrealCorpusTools/MFA-reorganization-scripts
for more details.
"""
import argparse
import os
from pathlib import Path
from typing import Union


def link_wav(root_dir: Union[str, Path], output_dir: Union[str, Path]):
    for sub_set in {'train', 'test'}:
        wav_dir = root_dir / sub_set / 'wav'
        new_dir = output_dir / sub_set
        new_dir.mkdir(parents=True, exist_ok=True)

        for spk_dir in os.listdir(wav_dir):
            sub_dir = wav_dir / spk_dir
            new_sub_dir = new_dir / spk_dir
            os.symlink(sub_dir, new_sub_dir)


def write_lab(root_dir: Union[str, Path],
              output_dir: Union[str, Path],
              script_type='pinyin'):
    for sub_set in {'train', 'test'}:
        text_path = root_dir / sub_set / 'content.txt'
        new_dir = output_dir / sub_set

        with open(text_path, 'r') as rf:
            for line in rf:
                wav_id, context = line.strip().split('\t')
                spk_id = wav_id[:7]
                transcript_name = wav_id.split('.')[0] + '.lab'
                transcript_path = new_dir / spk_id / transcript_name
                context_list = context.split()
                word_list = context_list[0:-1:2]
                pinyin_list = context_list[1::2]
                wf = open(transcript_path, 'w')
                if script_type == 'word':
                    # add space between chinese char
                    new_context = ' '.join(word_list)
                elif script_type == 'pinyin':
                    new_context = ' '.join(pinyin_list)
                wf.write(new_context + '\n')


def reorganize_aishell3(root_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        script_type='pinyin'):
    output_dir.mkdir(parents=True, exist_ok=True)
    link_wav(root_dir, output_dir)
    write_lab(root_dir, output_dir, script_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize AISHELL-3 dataset for MFA")
    parser.add_argument(
        "--root-dir", type=str, default="", help="path to AISHELL-3 dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to save outputs (audio and transcriptions)")
    parser.add_argument(
        "--script-type",
        type=str,
        default="pinyin",
        help="type of lab ('word'/'pinyin')")

    args = parser.parse_args()
    root_dir = Path(args.root_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    reorganize_aishell3(root_dir, output_dir, args.script_type)
