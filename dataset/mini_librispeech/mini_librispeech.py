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
import argparse
import codecs
import io
import json
import os
from multiprocessing.pool import Pool

import soundfile

from utils.utility import download
from utils.utility import unpack

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
    total_sec = 0.0
    total_text = 0.0
    total_num = 0

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

                utt = os.path.splitext(os.path.basename(audio_filepath))[0]
                utt2spk = '-'.join(utt.split('-')[:2])
                json_lines.append(
                    json.dumps({
                        'utt': utt,
                        'utt2spk': utt2spk,
                        'feat': audio_filepath,
                        'feat_shape': (duration, ),  #second
                        'text': text,
                    }))

                total_sec += duration
                total_text += len(text)
                total_num += 1

    with codecs.open(manifest_path, 'w', 'utf-8') as out_file:
        for line in json_lines:
            out_file.write(line + '\n')

    subset = os.path.splitext(manifest_path)[1][1:]
    manifest_dir = os.path.dirname(manifest_path)
    data_dir_name = os.path.split(data_dir)[-1]
    meta_path = os.path.join(manifest_dir, data_dir_name) + '.meta'
    with open(meta_path, 'w') as f:
        print(f"{subset}:", file=f)
        print(f"{total_num} utts", file=f)
        print(f"{total_sec / (60*60)} h", file=f)
        print(f"{total_text} text", file=f)
        print(f"{total_text / total_sec} text/sec", file=f)
        print(f"{total_sec / total_num} sec/utt", file=f)


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

    tasks = [
        (URL_TRAIN_CLEAN, MD5_TRAIN_CLEAN,
         os.path.join(args.target_dir, "train-clean"),
         args.manifest_prefix + ".train-clean"),
        (URL_DEV_CLEAN, MD5_DEV_CLEAN, os.path.join(
            args.target_dir, "dev-clean"), args.manifest_prefix + ".dev-clean"),
    ]

    with Pool(2) as pool:
        pool.starmap(prepare_dataset, tasks)

    print("Data download and manifest prepare done!")


if __name__ == '__main__':
    main()
