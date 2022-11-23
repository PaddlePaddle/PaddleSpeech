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

# all the data will be download in the current data/voxceleb directory default
DATA_HOME = os.path.expanduser('.')

# if you use the http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/ as the download base url
# you need to get the username & password via the google form

# if you use the https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a as the download base url,
# you need use --no-check-certificate to connect the target download url 

BASE_URL = "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a"

# dev data
DEV_LIST = {
    "vox1_dev_wav_partaa": "e395d020928bc15670b570a21695ed96",
    "vox1_dev_wav_partab": "bbfaaccefab65d82b21903e81a8a8020",
    "vox1_dev_wav_partac": "017d579a2a96a077f40042ec33e51512",
    "vox1_dev_wav_partad": "7bb1e9f70fddc7a678fa998ea8b3ba19",
}
DEV_TARGET_DATA = "vox1_dev_wav_parta* vox1_dev_wav.zip ae63e55b951748cc486645f532ba230b"

# test data
TEST_LIST = {"vox1_test_wav.zip": "185fdc63c3c739954633d50379a3d102"}
TEST_TARGET_DATA = "vox1_test_wav.zip vox1_test_wav.zip 185fdc63c3c739954633d50379a3d102"

# voxceleb trial

TRIAL_BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/"
TRIAL_LIST = {
    "veri_test.txt": "29fc7cc1c5d59f0816dc15d6e8be60f7",  # voxceleb1
    "veri_test2.txt": "b73110731c9223c1461fe49cb48dddfc",  # voxceleb1(cleaned)
    "list_test_hard.txt": "21c341b6b2168eea2634df0fb4b8fff1",  # voxceleb1-H
    "list_test_hard2.txt":
    "857790e09d579a68eb2e339a090343c8",  # voxceleb1-H(cleaned)
    "list_test_all.txt": "b9ecf7aa49d4b656aa927a8092844e4a",  # voxceleb1-E
    "list_test_all2.txt":
    "a53e059deb562ffcfc092bf5d90d9f3a"  # voxceleb1-E(cleaned)
}

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
    print(f"Creating manifest {manifest_path_prefix} from {data_dir}")
    json_lines = []
    data_path = os.path.join(data_dir, "wav", "**", "*.wav")
    total_sec = 0.0
    total_text = 0.0
    total_num = 0
    speakers = set()
    for audio_path in glob.glob(data_path, recursive=True):
        audio_id = "-".join(audio_path.split("/")[-3:])
        utt2spk = audio_path.split("/")[-3]
        duration = soundfile.info(audio_path).duration
        text = ""
        json_lines.append(
            json.dumps(
                {
                    "utt": audio_id,
                    "utt2spk": str(utt2spk),
                    "feat": audio_path,
                    "feat_shape": (duration, ),
                    "text": text  # compatible with asr data format
                },
                ensure_ascii=False))

        total_sec += duration
        total_text += len(text)
        total_num += 1
        speakers.add(utt2spk)

    # data_dir_name refer to dev or test
    # voxceleb1 is given explicit in the path
    data_dir_name = Path(data_dir).name
    manifest_path_prefix = manifest_path_prefix + "." + data_dir_name
    if not os.path.exists(os.path.dirname(manifest_path_prefix)):
        os.makedirs(os.path.dirname(manifest_path_prefix))

    with codecs.open(manifest_path_prefix, 'w', encoding='utf-8') as f:
        for line in json_lines:
            f.write(line + "\n")

    manifest_dir = os.path.dirname(manifest_path_prefix)
    meta_path = os.path.join(manifest_dir, "voxceleb1." +
                             data_dir_name) + ".meta"
    with codecs.open(meta_path, 'w', encoding='utf-8') as f:
        print(f"{total_num} utts", file=f)
        print(f"{len(speakers)} speakers", file=f)
        print(f"{total_sec / (60 * 60)} h", file=f)
        print(f"{total_text} text", file=f)
        print(f"{total_text / total_sec} text/sec", file=f)
        print(f"{total_sec / total_num} sec/utt", file=f)


def prepare_dataset(base_url, data_list, target_dir, manifest_path,
                    target_data):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # wav directory already exists, it need do nothing
    # we will download the voxceleb1 data to ${target_dir}/vox1/dev/ or ${target_dir}/vox1/test directory 
    if not os.path.exists(os.path.join(target_dir, "wav")):
        # download all dataset part
        print(f"start to download the vox1 zip package to {target_dir}")
        for zip_part in data_list.keys():
            download_url = " --no-check-certificate " + base_url + "/" + zip_part
            download(
                url=download_url,
                md5sum=data_list[zip_part],
                target_dir=target_dir)

        # pack the all part to target zip file
        all_target_part, target_name, target_md5sum = target_data.split()
        target_name = os.path.join(target_dir, target_name)
        if not os.path.exists(target_name):
            pack_part_cmd = "cat {}/{} > {}".format(target_dir, all_target_part,
                                                    target_name)
            subprocess.call(pack_part_cmd, shell=True)

        # check the target zip file md5sum
        if not check_md5sum(target_name, target_md5sum):
            raise RuntimeError("{} MD5 checkssum failed".format(target_name))
        else:
            print("Check {} md5sum successfully".format(target_name))

        # unzip the all zip file
        if target_name.endswith(".zip"):
            unzip(target_name, target_dir)

    # create the manifest file
    create_manifest(data_dir=target_dir, manifest_path_prefix=manifest_path)


def prepare_trial(base_url, data_list, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for trial, md5sum in data_list.items():
        target_trial = os.path.join(target_dir, trial)
        if not os.path.exists(os.path.join(target_dir, trial)):
            download_url = " --no-check-certificate " + base_url + "/" + trial
            download(url=download_url, md5sum=md5sum, target_dir=target_dir)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    # prepare the vox1 dev data
    prepare_dataset(
        base_url=BASE_URL,
        data_list=DEV_LIST,
        target_dir=os.path.join(args.target_dir, "dev"),
        manifest_path=args.manifest_prefix,
        target_data=DEV_TARGET_DATA)

    # prepare the vox1 test data
    prepare_dataset(
        base_url=BASE_URL,
        data_list=TEST_LIST,
        target_dir=os.path.join(args.target_dir, "test"),
        manifest_path=args.manifest_prefix,
        target_data=TEST_TARGET_DATA)

    # prepare the vox1 trial
    prepare_trial(
        base_url=TRIAL_BASE_URL,
        data_list=TRIAL_LIST,
        target_dir=os.path.dirname(args.manifest_prefix))

    print("Manifest prepare done!")


if __name__ == '__main__':
    main()
