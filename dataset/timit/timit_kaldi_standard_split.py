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
"""Prepare TIMIT dataset (Standard split from Kaldi)

Create manifest files from splited datased.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
import argparse
import codecs
import json
import os
from pathlib import Path

import soundfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--src_dir",
    default="",
    type=str,
    help="Directory to kaldi splited data. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()


def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % manifest_path_prefix)
    json_lines = []

    data_types = ['train', 'dev', 'test']
    for dtype in data_types:
        del json_lines[:]
        total_sec = 0.0
        total_text = 0.0
        total_num = 0

        phn_path = os.path.join(data_dir, dtype + '.text')
        phn_dict = {}
        for line in codecs.open(phn_path, 'r', 'utf-8'):
            line = line.strip()
            if line == '':
                continue
            audio_id, text = line.split(' ', 1)
            phn_dict[audio_id] = text

        audio_dir = os.path.join(data_dir, dtype + '_sph.scp')
        for line in codecs.open(audio_dir, 'r', 'utf-8'):
            audio_id, audio_path = line.strip().split()
            # if no transcription for audio then raise error
            assert audio_id in phn_dict
            audio_data, samplerate = soundfile.read(audio_path)
            duration = float(len(audio_data) / samplerate)
            text = phn_dict[audio_id]

            gender_spk = str(Path(audio_path).parent.stem)
            spk = gender_spk[1:]
            gender = gender_spk[0]
            utt_id = '_'.join([spk, gender, audio_id])
            json_lines.append(
                json.dumps(
                    {
                        'utt': audio_id,
                        'utt2spk': spk,
                        'utt2gender': gender,
                        'feat': audio_path,
                        'feat_shape': (duration, ),  # second
                        'text': text
                    },
                    ensure_ascii=False))

            total_sec += duration
            total_text += len(text)
            total_num += 1

        manifest_path = manifest_path_prefix + '.' + dtype + '.raw'
        with codecs.open(manifest_path, 'w', 'utf-8') as fout:
            for line in json_lines:
                fout.write(line + '\n')


def prepare_dataset(src_dir, manifest_path=None):
    """create manifest file."""
    if os.path.isdir(manifest_path):
        manifest_path = os.path.join(manifest_path, 'manifest')
    if manifest_path:
        create_manifest(src_dir, manifest_path)


def main():
    if args.src_dir.startswith('~'):
        args.src_dir = os.path.expanduser(args.src_dir)

    prepare_dataset(src_dir=args.src_dir, manifest_path=args.manifest_prefix)

    print("manifest prepare done!")


if __name__ == '__main__':
    main()
