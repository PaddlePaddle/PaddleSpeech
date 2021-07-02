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

from deepspeech.frontend.utility import read_manifest


def link_wav(root_dir: Union[str, Path], output_dir: Union[str, Path]):
    manifest_path = root_dir / "manifest.train"
    manifest_jsons = read_manifest(manifest_path)
    for line_json in manifest_jsons:
        wav_path = line_json['feat']
        wav_name = wav_path.split("/")[-1]
        new_wav_path = output_dir / wav_name
        os.symlink(wav_path, new_wav_path)


def link_lexicon(root_dir: Union[str, Path],
                 output_dir: Union[str, Path],
                 script_type='phone'):
    manifest_path = root_dir / "manifest.train"
    manifest_jsons = read_manifest(manifest_path)
    line_json = manifest_jsons[0]
    wav_path = line_json['feat']

    if script_type == 'phone':
        # find lexicon.txt in THCHS-30
        grader_father = os.path.abspath(
            os.path.dirname(wav_path) + os.path.sep + "..")
        grader_father = Path(grader_father).expanduser()
        lexicon_name = "lexicon.txt"
        lexicon_father_dir = "lm_phone"
        lexicon_path = grader_father / lexicon_father_dir / lexicon_name
    elif script_type == 'syllable':
        # find thchs30_pinyin2phone in dir of this py file
        py_dir_path = os.path.split(os.path.realpath(__file__))[0]
        py_dir_path = Path(py_dir_path).expanduser()
        lexicon_path = py_dir_path / "thchs30_pinyin2phone"
    else:
        # script_type == 'text'
        # find thchs30_cn2phone in dir of this py file
        py_dir_path = os.path.split(os.path.realpath(__file__))[0]
        py_dir_path = Path(py_dir_path).expanduser()
        lexicon_path = py_dir_path / "thchs30_cn2phone"

    new_lexicon_name = script_type + ".lexicon"
    new_lexicon_path = os.path.dirname(output_dir) + "/" + new_lexicon_name
    os.symlink(lexicon_path, new_lexicon_path)


def dump_lab(root_dir: Union[str, Path],
             output_dir: Union[str, Path],
             script_type='phone'):
    # script_type can in {'text', 'syllable', 'phone'}
    manifest_path = root_dir / "manifest.train"
    manifest_jsons = read_manifest(manifest_path)
    for line_json in manifest_jsons:
        utt_id = line_json['utt']
        transcript_name = utt_id + ".lab"
        transcript_path = output_dir / transcript_name
        with open(transcript_path, 'wt') as wf:
            wf.write(line_json[script_type] + "\n")


def reorganize_thchs30(root_dir: Union[str, Path],
                       output_dir: Union[str, Path]=None,
                       script_type='phone'):
    root_dir = Path(root_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    link_wav(root_dir, output_dir)
    dump_lab(root_dir, output_dir, script_type)
    link_lexicon(root_dir, output_dir, script_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize THCHS-30 dataset for MFA")
    parser.add_argument("--root-dir", type=str, help="path to thchs30 dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to save outputs(audio and transcriptions)")

    parser.add_argument(
        "--script-type",
        type=str,
        default="phone",
        help="type of lab (text'/'syllable'/'phone')")
    args = parser.parse_args()
    reorganize_thchs30(args.root_dir, args.output_dir, args.script_type)
