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
"""Prepare TALCS ASR datasets.

create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
import argparse
import codecs
import io
import json
import os

import soundfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()

TRAIN_SET = os.path.join(args.target_dir, "train_set")
DEV_SET = os.path.join(args.target_dir, "dev_set")
TEST_SET = os.path.join(args.target_dir, "test_set")

manifest_train_path = os.path.join(args.manifest_prefix, "manifest.train.raw")
manifest_dev_path = os.path.join(args.manifest_prefix, "manifest.dev.raw")
manifest_test_path = os.path.join(args.manifest_prefix, "manifest.test.raw")


def create_manifest(data_dir, manifest_path):
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. audio filepath, transcription text, audio
    duration) of each audio file within the data set.
    """
    print("Creating manifest %s ..." % manifest_path)
    json_lines = []
    total_sec = 0.0
    total_char = 0.0
    total_num = 0
    wav_dir = os.path.join(data_dir, 'wav')
    text_filepath = os.path.join(data_dir, 'label.txt')
    for subfolder, _, filelist in sorted(os.walk(wav_dir)):
        text_filelist = text_filepath
        if len(text_filelist) > 0:
            for line in io.open(text_filepath, encoding="utf8"):
                segments = line.strip().split()
                nchars = len(segments[1:])
                text = ' '.join(segments[1:]).lower()

                audio_filepath = os.path.abspath(
                    os.path.join(subfolder, segments[0] + '.wav'))
                audio_data, samplerate = soundfile.read(audio_filepath)
                duration = float(len(audio_data)) / samplerate

                utt = os.path.splitext(os.path.basename(audio_filepath))[0]
                utt2spk = '-'.join(utt.split('-')[:2])

                json_lines.append(
                    json.dumps({
                        'utt': utt,
                        'utt2spk': utt2spk,
                        'feat': audio_filepath,
                        'feat_shape': (duration, ),  # second
                        'text': text,
                    }))

                total_sec += duration
                total_char += nchars
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
        print(f"{total_char} char", file=f)
        print(f"{total_char / total_sec} char/sec", file=f)
        print(f"{total_sec / total_num} sec/utt", file=f)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    create_manifest(TRAIN_SET, manifest_train_path)
    create_manifest(DEV_SET, manifest_dev_path)
    create_manifest(TEST_SET, manifest_test_path)
    print("Data download and manifest prepare done!")


if __name__ == '__main__':
    main()
