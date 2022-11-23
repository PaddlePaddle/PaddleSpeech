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
"""Prepare aidatatang_200zh mandarin dataset

Download, unpack and create manifest files.
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

from utils.utility import download
from utils.utility import unpack

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

URL_ROOT = 'http://www.openslr.org/resources/62'
# URL_ROOT = 'https://openslr.magicdatatech.com/resources/62'
DATA_URL = URL_ROOT + '/aidatatang_200zh.tgz'
MD5_DATA = '6e0f4f39cd5f667a7ee53c397c8d0949'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/aidatatang_200zh",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()


def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % manifest_path_prefix)
    json_lines = []
    transcript_path = os.path.join(data_dir, 'transcript',
                                   'aidatatang_200_zh_transcript.txt')
    transcript_dict = {}
    for line in codecs.open(transcript_path, 'r', 'utf-8'):
        line = line.strip()
        if line == '':
            continue
        audio_id, text = line.split(' ', 1)
        # remove withespace, charactor text
        text = ''.join(text.split())
        transcript_dict[audio_id] = text

    data_types = ['train', 'dev', 'test']
    for dtype in data_types:
        del json_lines[:]
        total_sec = 0.0
        total_text = 0.0
        total_num = 0

        audio_dir = os.path.join(data_dir, 'corpus/', dtype)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for fname in filelist:
                if not fname.endswith('.wav'):
                    continue

                audio_path = os.path.abspath(os.path.join(subfolder, fname))
                audio_id = os.path.basename(fname)[:-4]
                utt2spk = Path(audio_path).parent.name

                audio_data, samplerate = soundfile.read(audio_path)
                duration = float(len(audio_data) / samplerate)
                text = transcript_dict[audio_id]
                json_lines.append(
                    json.dumps(
                        {
                            'utt': audio_id,
                            'utt2spk': str(utt2spk),
                            'feat': audio_path,
                            'feat_shape': (duration, ),  # second
                            'text': text,
                        },
                        ensure_ascii=False))

                total_sec += duration
                total_text += len(text)
                total_num += 1

        manifest_path = manifest_path_prefix + '.' + dtype
        with codecs.open(manifest_path, 'w', 'utf-8') as fout:
            for line in json_lines:
                fout.write(line + '\n')

        manifest_dir = os.path.dirname(manifest_path_prefix)
        meta_path = os.path.join(manifest_dir, dtype) + '.meta'
        with open(meta_path, 'w') as f:
            print(f"{dtype}:", file=f)
            print(f"{total_num} utts", file=f)
            print(f"{total_sec / (60*60)} h", file=f)
            print(f"{total_text} text", file=f)
            print(f"{total_text / total_sec} text/sec", file=f)
            print(f"{total_sec / total_num} sec/utt", file=f)


def prepare_dataset(url, md5sum, target_dir, manifest_path, subset):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, subset)
    if not os.path.exists(data_dir):
        filepath = download(url, md5sum, target_dir)
        unpack(filepath, target_dir)
        # unpack all audio tar files
        audio_dir = os.path.join(data_dir, 'corpus')
        for subfolder, dirlist, filelist in sorted(os.walk(audio_dir)):
            for sub in dirlist:
                print(f"unpack dir {sub}...")
                for folder, _, filelist in sorted(
                        os.walk(os.path.join(subfolder, sub))):
                    for ftar in filelist:
                        unpack(os.path.join(folder, ftar), folder, True)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)

    create_manifest(data_dir, manifest_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        url=DATA_URL,
        md5sum=MD5_DATA,
        target_dir=args.target_dir,
        manifest_path=args.manifest_prefix,
        subset='aidatatang_200zh')

    print("Data download and manifest prepare done!")


if __name__ == '__main__':
    main()
