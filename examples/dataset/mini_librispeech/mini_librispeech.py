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
"""Prepare Librispeech ASR datasets.

Download, unpack and create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""

import distutils.util
import os
import sys
import argparse
import soundfile
import json
import codecs
import io
from utils.utility import download, unpack

URL_ROOT = "http://www.openslr.org/resources/31"
URL_TRAIN_CLEAN = URL_ROOT + "/train-clean-5.tar.gz"
URL_DEV_CLEAN = URL_ROOT + "/dev-clean-2.tar.gz"

MD5_TRAIN_CLEAN = "5df7d4e78065366204ca6845bb08f490"
MD5_DEV_CLEAN = "6d7ab67ac6a1d2c993d050e16d61080d"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default='~/.cache/paddle/dataset/speech/libri',
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()


def create_manifest(data_dir, manifest_path):
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. audio filepath, transcription text, audio
    duration) of each audio file within the data set.
    """
    print("Creating manifest %s ..." % manifest_path)
    json_lines = []
    for subfolder, _, filelist in sorted(os.walk(data_dir)):
        text_filelist = [
            filename for filename in filelist if filename.endswith('trans.txt')
        ]
        if len(text_filelist) > 0:
            text_filepath = os.path.join(subfolder, text_filelist[0])
            for line in io.open(text_filepath, encoding="utf8"):
                segments = line.strip().split()
                text = ' '.join(segments[1:]).lower()
                audio_filepath = os.path.join(subfolder, segments[0] + '.flac')
                audio_data, samplerate = soundfile.read(audio_filepath)
                duration = float(len(audio_data)) / samplerate
                json_lines.append(
                    json.dumps({
                        'utt':
                        os.path.splitext(os.path.basename(audio_filepath))[0],
                        'feat':
                        audio_filepath,
                        'feat_shape': (duration, ),  #second
                        'text':
                        text
                    }))
    with codecs.open(manifest_path, 'w', 'utf-8') as out_file:
        for line in json_lines:
            out_file.write(line + '\n')


def prepare_dataset(url, md5sum, target_dir, manifest_path):
    """Download, unpack and create summmary manifest file.
    """
    if not os.path.exists(os.path.join(target_dir, "LibriSpeech")):
        # download
        filepath = download(url, md5sum, target_dir)
        # unpack
        unpack(filepath, target_dir)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)
    # create manifest json file
    create_manifest(target_dir, manifest_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        url=URL_TRAIN_CLEAN,
        md5sum=MD5_TRAIN_CLEAN,
        target_dir=os.path.join(args.target_dir, "train-clean"),
        manifest_path=args.manifest_prefix + ".train-clean")
    prepare_dataset(
        url=URL_DEV_CLEAN,
        md5sum=MD5_DEV_CLEAN,
        target_dir=os.path.join(args.target_dir, "dev-clean"),
        manifest_path=args.manifest_prefix + ".dev-clean")


if __name__ == '__main__':
    main()
