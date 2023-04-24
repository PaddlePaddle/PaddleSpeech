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
import argparse
import codecs
import json
import os
from pathlib import Path

import soundfile

from paddlespeech.dataset.download import download
from paddlespeech.dataset.download import unpack
from paddlespeech.utils.argparse import print_arguments

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

URL_ROOT = 'http://openslr.elda.org/resources/33'
# URL_ROOT = 'https://openslr.magicdatatech.com/resources/33'
DATA_URL = URL_ROOT + '/data_aishell.tgz'
MD5_DATA = '2f494334227864a8a8fec932999db9d8'
RESOURCE_URL = URL_ROOT + '/resource_aishell.tgz'
MD5_RESOURCE = '957d480a0fcac85fc18e550756f624e5'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/Aishell",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()


def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % os.path.join(data_dir,
                                                    manifest_path_prefix))
    json_lines = []
    transcript_path = os.path.join(data_dir, 'transcript',
                                   'aishell_transcript_v0.8.txt')
    transcript_dict = {}
    for line in codecs.open(transcript_path, 'r', 'utf-8'):
        line = line.strip()
        if line == '':
            continue
        audio_id, text = line.split(' ', 1)
        # remove withespace, charactor text
        text = ''.join(text.split())
        transcript_dict[audio_id] = text

    data_metas = dict()
    data_types = ['train', 'dev', 'test']
    for dtype in data_types:
        del json_lines[:]
        total_sec = 0.0
        total_text = 0.0
        total_num = 0

        audio_dir = os.path.join(data_dir, 'wav', dtype)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for fname in filelist:
                audio_path = os.path.abspath(os.path.join(subfolder, fname))
                audio_id = os.path.basename(fname)[:-4]
                # if no transcription for audio then skipped
                if audio_id not in transcript_dict:
                    continue

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
                            'text': text
                        },
                        ensure_ascii=False))

                total_sec += duration
                total_text += len(text)
                total_num += 1

        manifest_path = manifest_path_prefix + '.' + dtype
        with codecs.open(manifest_path, 'w', 'utf-8') as fout:
            for line in json_lines:
                fout.write(line + '\n')

        meta = dict()
        meta["dtype"] = dtype  # train, dev, test
        meta["utts"] = total_num
        meta["hours"] = total_sec / (60 * 60)
        meta["text"] = total_text
        meta["text/sec"] = total_text / total_sec
        meta["sec/utt"] = total_sec / total_num
        data_metas[dtype] = meta

        manifest_dir = os.path.dirname(manifest_path_prefix)
        meta_path = os.path.join(manifest_dir, dtype) + '.meta'
        with open(meta_path, 'w') as f:
            for key, val in meta.items():
                print(f"{key}: {val}", file=f)

    return data_metas


def download_dataset(url, md5sum, target_dir):
    """Download, unpack and create manifest file."""
    data_dir = os.path.join(target_dir, 'data_aishell')
    if not os.path.exists(data_dir):
        filepath = download(url, md5sum, target_dir)
        unpack(filepath, target_dir)
        # unpack all audio tar files
        audio_dir = os.path.join(data_dir, 'wav')
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for ftar in filelist:
                unpack(os.path.join(subfolder, ftar), subfolder, True)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              os.path.abspath(target_dir))
    return os.path.abspath(data_dir)


def check_dataset(data_dir):
    print(f"check dataset {os.path.abspath(data_dir)} ...")

    transcript_path = os.path.join(data_dir, 'transcript',
                                   'aishell_transcript_v0.8.txt')
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"no transcript file found in {data_dir}.")

    transcript_dict = {}
    for line in codecs.open(transcript_path, 'r', 'utf-8'):
        line = line.strip()
        if line == '':
            continue
        audio_id, text = line.split(' ', 1)
        # remove withespace, charactor text
        text = ''.join(text.split())
        transcript_dict[audio_id] = text

    no_label = 0
    data_types = ['train', 'dev', 'test']
    for dtype in data_types:
        audio_dir = os.path.join(data_dir, 'wav', dtype)
        if not os.path.exists(audio_dir):
            raise IOError(f"{audio_dir} does not exist.")

        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for fname in filelist:
                audio_path = os.path.abspath(os.path.join(subfolder, fname))
                audio_id = os.path.basename(fname)[:-4]
                # if no transcription for audio then skipped
                if audio_id not in transcript_dict:
                    print(f"Warning: {audio_id} not has transcript.")
                    no_label += 1
                    continue

                utt2spk = Path(audio_path).parent.name
                audio_data, samplerate = soundfile.read(audio_path)
                assert samplerate == 16000, f"{audio_path} sample rate is {samplerate} not 16k, please check."

        print(f"Warning: {dtype} has {no_label} audio does not has transcript.")


def prepare_dataset(url, md5sum, target_dir, manifest_path=None, check=False):
    """Download, unpack and create manifest file."""
    data_dir = download_dataset(url, md5sum, target_dir)

    if check:
        try:
            check_dataset(data_dir)
        except Exception as e:
            raise ValueError(
                f"{data_dir} dataset format not right, please check it.")

    meta = None
    if manifest_path:
        meta = create_manifest(data_dir, manifest_path)

    return data_dir, meta


def main():
    print_arguments(args, globals())
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    data_dir, meta = prepare_dataset(
        url=DATA_URL,
        md5sum=MD5_DATA,
        target_dir=args.target_dir,
        manifest_path=args.manifest_prefix,
        check=True)

    resource_dir, _ = prepare_dataset(
        url=RESOURCE_URL,
        md5sum=MD5_RESOURCE,
        target_dir=args.target_dir,
        manifest_path=None)

    print("Data download and manifest prepare done!")


if __name__ == '__main__':
    main()
