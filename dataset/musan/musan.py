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
"""Prepare Aishell mandarin dataset

Download, unpack and create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import json
import os

import soundfile

from utils.utility import download
from utils.utility import unpack

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

URL_ROOT = 'https://www.openslr.org/resources/17'
DATA_URL = URL_ROOT + '/musan.tar.gz'
MD5_DATA = '0c472d4fc0c5141eca47ad1ffeb2a7df'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/musan",
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
    data_types = ['music', 'noise', 'speech']
    for dtype in data_types:
        del json_lines[:]
        audio_dir = os.path.join(data_dir, dtype)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            print('x, ', subfolder)
            for fname in filelist:
                audio_path = os.path.join(subfolder, fname)
                if not audio_path.endswith('.wav'):
                    continue
                audio_data, samplerate = soundfile.read(audio_path)
                duration = float(len(audio_data) / samplerate)
                json_lines.append(
                    json.dumps(
                        {
                            'utt':
                            os.path.splitext(os.path.basename(audio_path))[0],
                            'feat':
                            audio_path,
                            'feat_shape': (duration, ),  #second
                            'type':
                            dtype,
                        },
                        ensure_ascii=False))
        manifest_path = manifest_path_prefix + '.' + dtype
        with codecs.open(manifest_path, 'w', 'utf-8') as fout:
            for line in json_lines:
                fout.write(line + '\n')


def prepare_dataset(url, md5sum, target_dir, manifest_path):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'musan')
    if not os.path.exists(data_dir):
        filepath = download(url, md5sum, target_dir)
        unpack(filepath, target_dir)
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
        manifest_path=args.manifest_prefix)


if __name__ == '__main__':
    main()
