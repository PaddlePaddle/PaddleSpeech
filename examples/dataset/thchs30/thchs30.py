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
"""Prepare THCHS-30 mandarin dataset

Download, unpack and create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
import argparse
import codecs
import json
import os
from multiprocessing.pool import Pool
from pathlib import Path

import soundfile

from utils.utility import download
from utils.utility import unpack

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

URL_ROOT = 'http://www.openslr.org/resources/18'
# URL_ROOT = 'https://openslr.magicdatatech.com/resources/18'
DATA_URL = URL_ROOT + '/data_thchs30.tgz'
TEST_NOISE_URL = URL_ROOT + '/test-noise.tgz'
RESOURCE_URL = URL_ROOT + '/resource.tgz'
MD5_DATA = '2d2252bde5c8429929e1841d4cb95e90'
MD5_TEST_NOISE = '7e8a985fb965b84141b68c68556c2030'
MD5_RESOURCE = 'c0b2a565b4970a0c4fe89fefbf2d97e1'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/THCHS30",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()


def read_trn(filepath):
    """read trn file.
    word text in first line.
    syllable text in second line.
    phoneme text in third line.

    Args:
        filepath (str): trn path.

    Returns:
        list(str): (word, syllable, phone)
    """
    texts = []
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
        assert len(lines) == 3, lines
    # charactor text, remove withespace
    texts.append(''.join(lines[0].split()))
    texts.extend(lines[1:])
    return texts


def resolve_symlink(filepath):
    """resolve symlink which content is norm file.

    Args:
        filepath (str): norm file symlink.
    """
    sym_path = Path(filepath)
    relative_link = sym_path.read_text().strip()
    relative = Path(relative_link)
    relpath = sym_path.parent / relative
    return relpath.resolve()


def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % manifest_path_prefix)
    json_lines = []
    data_types = ['train', 'dev', 'test']
    for dtype in data_types:
        del json_lines[:]
        total_sec = 0.0
        total_text = 0.0
        total_num = 0

        audio_dir = os.path.join(data_dir, dtype)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for fname in filelist:
                file_path = os.path.join(subfolder, fname)
                if file_path.endswith('.wav'):
                    audio_path = os.path.abspath(file_path)
                    text_path = resolve_symlink(audio_path + '.trn')
                else:
                    continue

                assert os.path.exists(audio_path) and os.path.exists(text_path)

                audio_id = os.path.basename(audio_path)[:-4]
                spk = audio_id.split('_')[0]

                word_text, syllable_text, phone_text = read_trn(text_path)
                audio_data, samplerate = soundfile.read(audio_path)
                duration = float(len(audio_data) / samplerate)

                # not dump alignment infos
                json_lines.append(
                    json.dumps(
                        {
                            'utt': audio_id,
                            'utt2spk': spk,
                            'feat': audio_path,
                            'feat_shape': (duration, ),  # second
                            'text': word_text,  # charactor
                            'syllable': syllable_text,
                            'phone': phone_text,
                        },
                        ensure_ascii=False))

                total_sec += duration
                total_text += len(word_text)
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
    datadir = os.path.join(target_dir, subset)
    if not os.path.exists(datadir):
        filepath = download(url, md5sum, target_dir)
        unpack(filepath, target_dir)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)

    if subset == 'data_thchs30':
        create_manifest(datadir, manifest_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    tasks = [
        (DATA_URL, MD5_DATA, args.target_dir, args.manifest_prefix,
         "data_thchs30"),
        (TEST_NOISE_URL, MD5_TEST_NOISE, args.target_dir, args.manifest_prefix,
         "test-noise"),
        (RESOURCE_URL, MD5_RESOURCE, args.target_dir, args.manifest_prefix,
         "resource"),
    ]
    with Pool(7) as pool:
        pool.starmap(prepare_dataset, tasks)

    print("Data download and manifest prepare done!")


if __name__ == '__main__':
    main()
