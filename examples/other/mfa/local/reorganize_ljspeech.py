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
"""Script to reorganize LJSpeech-1.1 dataset so as to use Montreal Force
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
    wav_dir = root_dir / 'wavs'
    for spk_dir in os.listdir(wav_dir):
        sub_dir = wav_dir / spk_dir
        new_sub_dir = output_dir / spk_dir
        os.symlink(sub_dir, new_sub_dir)


def write_lab(root_dir: Union[str, Path], output_dir: Union[str, Path]):

    text_path = root_dir / 'metadata.csv'
    with open(text_path, 'r') as rf:
        for line in rf:
            line_list = line.strip().split('|')
            utt = line_list[0]
            raw_text = line_list[-1]
            transcript_name = utt + '.lab'
            transcript_path = output_dir / transcript_name
            with open(transcript_path, 'w') as wf:
                wf.write(raw_text + '\n')


def reorganize_ljspeech(root_dir: Union[str, Path], output_dir: Union[str,
                                                                      Path]):

    link_wav(root_dir, output_dir)
    write_lab(root_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize LJSpeech-1.1 dataset for MFA")
    parser.add_argument("--root-dir",
                        type=str,
                        help="path to LJSpeech-1.1 dataset.")
    parser.add_argument("--output-dir",
                        type=str,
                        help="path to save outputs (audio and transcriptions)")
    args = parser.parse_args()
    root_dir = Path(args.root_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    reorganize_ljspeech(root_dir, output_dir)
