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
"""Prepare Ted-En-Zh speech translation dataset

Create manifest files from splited datased. 
dev set: tst2010, test set: tst2015
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
import argparse
import codecs
import json
import os

import soundfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--src-dir",
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

    data_types_infos = [
        ('train', 'train-split/train-segment', 'En-Zh/train.en-zh'),
        ('dev', 'test-segment/tst2010', 'En-Zh/tst2010.en-zh'),
        ('test', 'test-segment/tst2015', 'En-Zh/tst2015.en-zh')
    ]
    for data_info in data_types_infos:
        dtype, audio_relative_dir, text_relative_path = data_info
        del json_lines[:]
        total_sec = 0.0
        total_text = 0.0
        total_num = 0

        text_path = os.path.join(data_dir, text_relative_path)
        audio_dir = os.path.join(data_dir, audio_relative_dir)

        for line in codecs.open(text_path, 'r', 'utf-8', errors='ignore'):
            line = line.strip()
            if len(line) < 1:
                continue
            audio_id, trancription, translation = line.split('\t')
            utt = audio_id.split('.')[0]

            audio_path = os.path.join(audio_dir, audio_id)
            if os.path.exists(audio_path):
                if os.path.getsize(audio_path) < 30000:
                    continue
                audio_data, samplerate = soundfile.read(audio_path)
                duration = float(len(audio_data) / samplerate)

                translation_str = " ".join(translation.split())
                trancription_str = " ".join(trancription.split())
                json_lines.append(
                    json.dumps(
                        {
                            'utt': utt,
                            'feat': audio_path,
                            'feat_shape': (duration, ),  # second
                            'text': [translation_str, trancription_str],
                        },
                        ensure_ascii=False))

                total_sec += duration
                total_text += len(translation.split())
                total_num += 1
                if not total_num % 1000:
                    print(dtype, 'Processed:', total_num)

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
