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
"""Prepare VoxCeleb1 dataset

create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.

researchers should download the voxceleb1 dataset yourselves
through google form to get the username & password and unpack the data
"""
import argparse
import codecs
import glob
import json
import os
import subprocess
from pathlib import Path

import soundfile

from utils.utility import check_md5sum
from utils.utility import download
from utils.utility import unzip

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech/voxceleb/')

# if you use the http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/ as the download base url
# you need to get the username & password via the google form

# if you use the https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a as the download base url,
# you need use --no-check-certificate to connect the target download url 

BASE_URL = "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a"
DATA_LIST = {
    "vox1_dev_wav_partaa": "e395d020928bc15670b570a21695ed96",
    "vox1_dev_wav_partab": "bbfaaccefab65d82b21903e81a8a8020",
    "vox1_dev_wav_partac": "017d579a2a96a077f40042ec33e51512",
    "vox1_dev_wav_partad": "7bb1e9f70fddc7a678fa998ea8b3ba19",
    "vox1_test_wav.zip": "vox1_test_wav.zip",
}

TARGET_DATA = "vox1_dev_wav_parta* vox1_dev_wav.zip ae63e55b951748cc486645f532ba230b"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/voxceleb1/",
    type=str,
    help="Directory to save the voxceleb1 dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")

args = parser.parse_args()


def create_manifest(data_dir, manifest_path_prefix):
    print("Creating manifest %s ..." % manifest_path_prefix)
    json_lines = []
    data_path = os.path.join(data_dir, "wav", "**", "*.wav")
    total_sec = 0.0
    total_text = 0.0
    total_num = 0
    spkers = set()
    for audio_path in glob.glob(data_path, recursive=True):
        audio_id = "/".join(audio_path.split("/")[-3:])
        utt2spk = audio_path.split("/")[-3]
        duration = soundfile.info(audio_path).duration
        text = ""
        json_lines.append(
            json.dumps(
                {
                    "utt": audio_id,
                    "utt2spk": str(utt2spk),
                    "feat": audio_path,
                    "text_shape": (duration, ),
                    "text": text  # compatible with asr data format
                },
                ensure_ascii=False,
                indent=4))

        total_sec += duration
        total_text += len(text)
        total_num += 1
        spkers.add(utt2spk)

    with codecs.open(manifest_path_prefix, 'w', encoding='utf-8') as f:
        for line in json_lines:
            f.write(line + "\n")

    manifest_dir = os.path.dirname(manifest_path_prefix)
    # data_dir_name refer to voxceleb1, which is used to distingush the voxceleb2 dataset info
    data_dir_name = Path(data_dir).name
    meta_path = os.path.join(manifest_dir, data_dir_name) + ".meta"
    with codecs.open(meta_path, 'w', encoding='utf-8') as f:
        print(f"{total_num} utts", file=f)
        print(f"{len(spkers)} spkers", file=f)
        print(f"{total_sec / (60 * 60)} h", file=f)
        print(f"{total_text} text", file=f)
        print(f"{total_text / total_sec} text/sec", file=f)
        print(f"{total_sec / total_num} sec/utt", file=f)


def prepare_dataset(base_url, data_list, target_dir, manifest_path,
                    target_data):
    data_dir = os.path.join(target_dir, "voxceleb1")
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # download all dataset part
    for zip_part in data_list.keys():
        download_url = base_url + "/" + zip_part + " --no-check-certificate "
        download(
            url=download_url, md5sum=data_list[zip_part], target_dir=target_dir)

    # pack the all part to target zip file 
    all_target_part, target_name, target_md5sum = target_data.split()
    pack_part_cmd = "cat {}/{} > {}/{}".format(target_dir, all_target_part,
                                               target_dir, target_name)
    subprocess.call(pack_part_cmd, shell=True)

    # check the target zip file md5sum
    target_name = os.path.join(target_dir, target_name)
    if not check_md5sum(target_name, target_md5sum):
        raise RuntimeError("{} MD5 checkssum failed".format(target_name))

    # unzip the all zip file
    unzip(target_name, target_dir)
    unzip(os.path.join(target_dir, "vox1_test_wav.zip"), target_dir)

    # create the manifest file
    create_manifest(
        data_dir=args.target_dir, manifest_path_prefix=args.manifest_prefix)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        base_url=BASE_URL,
        data_list=DATA_LIST,
        target_dir=args.target_dir,
        manifest_path=args.manifest_prefix,
        target_data=TARGET_DATA)

    print("Manifest prepare done!")


if __name__ == '__main__':
    main()
