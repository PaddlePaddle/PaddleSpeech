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

import distutils.util
import soundfile

from paddlespeech.dataset.download import download
from paddlespeech.dataset.download import unpack
from paddlespeech.utils.argparse import strtobool

URL_ROOT = "http://openslr.elda.org/resources/12"
#URL_ROOT = "https://openslr.magicdatatech.com/resources/12"
URL_TEST_CLEAN = URL_ROOT + "/test-clean.tar.gz"
URL_TEST_OTHER = URL_ROOT + "/test-other.tar.gz"
URL_DEV_CLEAN = URL_ROOT + "/dev-clean.tar.gz"
URL_DEV_OTHER = URL_ROOT + "/dev-other.tar.gz"
URL_TRAIN_CLEAN_100 = URL_ROOT + "/train-clean-100.tar.gz"
URL_TRAIN_CLEAN_360 = URL_ROOT + "/train-clean-360.tar.gz"
URL_TRAIN_OTHER_500 = URL_ROOT + "/train-other-500.tar.gz"

MD5_TEST_CLEAN = "32fa31d27d2e1cad72775fee3f4849a9"
MD5_TEST_OTHER = "fb5a50374b501bb3bac4815ee91d3135"
MD5_DEV_CLEAN = "42e2234ba48799c1f50f24a7926300a1"
MD5_DEV_OTHER = "c8d0bcc9cca99d4f8b62fcc847357931"
MD5_TRAIN_CLEAN_100 = "2a93770f6d5c6c964bc36631d331a522"
MD5_TRAIN_CLEAN_360 = "c0e676e450a7ff2f54aeade5171606fa"
MD5_TRAIN_OTHER_500 = "d1a0fd59409feb2c614ce4d30c387708"

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
parser.add_argument(
    "--full_download",
    default="True",
    type=strtobool,
    help="Download all datasets for Librispeech."
    " If False, only download a minimal requirement (test-clean, dev-clean"
    " train-clean-100). (default: %(default)s)")
args = parser.parse_args()


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

    for subfolder, _, filelist in sorted(os.walk(data_dir)):
        text_filelist = [
            filename for filename in filelist if filename.endswith('trans.txt')
        ]
        if len(text_filelist) > 0:
            text_filepath = os.path.join(subfolder, text_filelist[0])
            for line in io.open(text_filepath, encoding="utf8"):
                segments = line.strip().split()
                nchars = len(segments[1:])
                text = ' '.join(segments[1:]).lower()

                audio_filepath = os.path.abspath(
                    os.path.join(subfolder, segments[0] + '.flac'))
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
        (URL_TEST_CLEAN, MD5_TEST_CLEAN, os.path.join(args.target_dir,
                                                      "test-clean"),
         args.manifest_prefix + ".test-clean"),
        (URL_DEV_CLEAN, MD5_DEV_CLEAN, os.path.join(
            args.target_dir, "dev-clean"), args.manifest_prefix + ".dev-clean"),
    ]
    if args.full_download:
        tasks.extend([
            (URL_TRAIN_CLEAN_100, MD5_TRAIN_CLEAN_100,
             os.path.join(args.target_dir, "train-clean-100"),
             args.manifest_prefix + ".train-clean-100"),
            (URL_TEST_OTHER, MD5_TEST_OTHER, os.path.join(args.target_dir,
                                                          "test-other"),
             args.manifest_prefix + ".test-other"),
            (URL_DEV_OTHER, MD5_DEV_OTHER, os.path.join(args.target_dir,
                                                        "dev-other"),
             args.manifest_prefix + ".dev-other"),
            (URL_TRAIN_CLEAN_360, MD5_TRAIN_CLEAN_360,
             os.path.join(args.target_dir, "train-clean-360"),
             args.manifest_prefix + ".train-clean-360"),
            (URL_TRAIN_OTHER_500, MD5_TRAIN_OTHER_500,
             os.path.join(args.target_dir, "train-other-500"),
             args.manifest_prefix + ".train-other-500"),
        ])

    with Pool(7) as pool:
        pool.starmap(prepare_dataset, tasks)

    print("Data download and manifest prepare done!")


if __name__ == '__main__':
    main()
