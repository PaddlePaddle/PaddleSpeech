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
"""Prepare VoxCeleb2 dataset

Download and unpack the voxceleb2 data files.
Voxceleb2 data is stored as the m4a format, 
so we need convert the m4a to wav with the convert.sh scripts
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

BASE_URL = "--no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data/"

# dev data
DEV_LIST = {
    "vox2_dev_aac_partaa": "da070494c573e5c0564b1d11c3b20577",
    "vox2_dev_aac_partab": "17fe6dab2b32b48abaf1676429cdd06f",
    "vox2_dev_aac_partac": "1de58e086c5edf63625af1cb6d831528",
    "vox2_dev_aac_partad": "5a043eb03e15c5a918ee6a52aad477f9",
    "vox2_dev_aac_partae": "cea401b624983e2d0b2a87fb5d59aa60",
    "vox2_dev_aac_partaf": "fc886d9ba90ab88e7880ee98effd6ae9",
    "vox2_dev_aac_partag": "d160ecc3f6ee3eed54d55349531cb42e",
    "vox2_dev_aac_partah": "6b84a81b9af72a9d9eecbb3b1f602e65",
}

DEV_TARGET_DATA = "vox2_dev_aac_parta* vox2_dev_aac.zip bbc063c46078a602ca71605645c2a402"

# test data
TEST_LIST = {"vox2_test_aac.zip": "0d2b3ea430a821c33263b5ea37ede312"}
TEST_TARGET_DATA = "vox2_test_aac.zip vox2_test_aac.zip 0d2b3ea430a821c33263b5ea37ede312"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/voxceleb2/",
    type=str,
    help="Directory to save the voxceleb1 dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
parser.add_argument(
    "--download",
    default=False,
    action="store_true",
    help="Download the voxceleb2 dataset. (default: %(default)s)")
parser.add_argument(
    "--generate",
    default=False,
    action="store_true",
    help="Generate the manifest files. (default: %(default)s)")

args = parser.parse_args()


def create_manifest(data_dir, manifest_path_prefix):
    """Generate the voxceleb2 dataset manifest file.
    We will create the ${manifest_path_prefix}.vox2 as the final manifest file 
    The dev and test wav info will be put in one manifest file.

    Args:
        data_dir (str): voxceleb2 wav directory, which include dev and test subdataset
        manifest_path_prefix (str): manifest file prefix
    """
    print("Creating manifest %s ..." % manifest_path_prefix)
    json_lines = []
    data_path = os.path.join(data_dir, "**", "*.wav")
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
    # voxceleb2 is given explicit in the path
    data_dir_name = Path(data_dir).name
    manifest_path_prefix = manifest_path_prefix + "." + data_dir_name

    if not os.path.exists(os.path.dirname(manifest_path_prefix)):
        os.makedirs(os.path.dirname(manifest_path_prefix))
    with codecs.open(manifest_path_prefix, 'w', encoding='utf-8') as f:
        for line in json_lines:
            f.write(line + "\n")

    manifest_dir = os.path.dirname(manifest_path_prefix)
    meta_path = os.path.join(manifest_dir, "voxceleb2." +
                             data_dir_name) + ".meta"
    with codecs.open(meta_path, 'w', encoding='utf-8') as f:
        print(f"{total_num} utts", file=f)
        print(f"{len(speakers)} speakers", file=f)
        print(f"{total_sec / (60 * 60)} h", file=f)
        print(f"{total_text} text", file=f)
        print(f"{total_text / total_sec} text/sec", file=f)
        print(f"{total_sec / total_num} sec/utt", file=f)


def download_dataset(base_url, data_list, target_data, target_dir, dataset):
    """Download the voxceleb2 zip package

    Args:
        base_url (str): the voxceleb2 dataset download baseline url
        data_list (dict): the dataset part zip package and the md5 value
        target_data (str): the final dataset zip info
        target_dir (str): the dataset stored directory
        dataset (str): the dataset name, dev or test

    Raises:
        RuntimeError: the md5sum occurs error
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # wav directory already exists, it need do nothing
    print("target dir {}".format(os.path.join(target_dir, dataset)))
    # unzip the dev dataset will create the dev and unzip the m4a to dev dir
    # but the test dataset will unzip to aac
    # so, wo create the ${target_dir}/test and unzip the m4a to test dir
    if not os.path.exists(os.path.join(target_dir, dataset)):
        print(f"start to download the vox2 zip package to {target_dir}")
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

        if dataset == "test":
            # we need make the test directory
            unzip(target_name, os.path.join(target_dir, "test"))
        else:
            # upzip dev zip pacakge and will create the dev directory
            unzip(target_name, target_dir)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    # download and unpack the vox2-dev data
    print("download: {}".format(args.download))
    if args.download:
        download_dataset(
            base_url=BASE_URL,
            data_list=DEV_LIST,
            target_data=DEV_TARGET_DATA,
            target_dir=args.target_dir,
            dataset="dev")

        download_dataset(
            base_url=BASE_URL,
            data_list=TEST_LIST,
            target_data=TEST_TARGET_DATA,
            target_dir=args.target_dir,
            dataset="test")

        print("VoxCeleb2 download is done!")

    if args.generate:
        create_manifest(
            args.target_dir, manifest_path_prefix=args.manifest_prefix)


if __name__ == '__main__':
    main()
