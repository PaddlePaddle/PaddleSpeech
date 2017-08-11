"""This tool is used for preparing data for DeepSpeech2 trainning on paddle cloud.

Steps:
1. Read original manifest and get the local path of sound files.
2. Tar all local sound files into one tar file.
3. Modify original manifest to remove the local path information.

Finally, we will get a tar file and a manifest with sound file name, duration
and text.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import tarfile
import sys
import argparse
import shutil
sys.path.append('../')
from data_utils.utils import read_manifest
from subprocess import call

TRAIN_TAR = "cloud.train.tar"
TRAIN_MANIFEST = "cloud.train.manifest"
TEST_TAR = "cloud.test.tar"
TEST_MANIFEST = "cloud.test.manifest"
VOCAB_FILE = "vocab.txt"
MEAN_STD_FILE = "mean_std.npz"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--train_manifest_path",
    default="../datasets/manifest.train",
    type=str,
    help="Manifest file of train data. (default: %(default)s)")
parser.add_argument(
    "--test_manifest_path",
    default="../datasets/manifest.test",
    type=str,
    help="Manifest file of test data. (default: %(default)s)")
parser.add_argument(
    "--vocab_file",
    default="../datasets/vocab/eng_vocab.txt",
    type=str,
    help="Vocab file to be uploaded to paddlecloud. (default: %(default)s)")
parser.add_argument(
    "--mean_std_file",
    default="../mean_std.npz",
    type=str,
    help="mean_std file to be uploaded to paddlecloud. (default: %(default)s)")
parser.add_argument(
    "--cloud_data_path",
    required=True,
    type=str,
    help="Destination path on  paddlecloud. (default: %(default)s)")
args = parser.parse_args()

parser.add_argument(
    "--local_tmp_path",
    default="./tmp/",
    type=str,
    help="Local directory for storing temporary  data. (default: %(default)s)")
args = parser.parse_args()


def pack_data(manifest_path, out_tar_path, out_manifest_path):
    '''1. According to the manifest, tar sound files into out_tar_path
    2. Generate a new manifest for output tar file
    '''
    out_tar = tarfile.open(out_tar_path, 'w')
    manifest = read_manifest(manifest_path)
    results = []
    for json_data in manifest:
        sound_file = json_data['audio_filepath']
        filename = os.path.basename(sound_file)
        out_tar.add(sound_file, arcname=filename)
        json_data['audio_filepath'] = filename
        results.append("%s\n" % json.dumps(json_data))
    with open(out_manifest_path, 'w') as out_manifest:
        out_manifest.writelines(results)
    out_manifest.close()
    out_tar.close()


def pcloud_cp(src, dst):
    """Copy src from local filesytem to dst in PaddleCloud filesystem.
    """
    ret = call(['paddlecloud', 'cp', src, dst])
    return ret


def pcloud_exist(path):
    """Check if file or directory exists in PaddleCloud filesystem.
    """
    ret = call(['paddlecloud', 'ls', path])
    return ret


if __name__ == '__main__':
    cloud_train_manifest = os.path.join(args.cloud_data_path, TRAIN_MANIFEST)
    cloud_train_tar = os.path.join(args.cloud_data_path, TRAIN_TAR)
    cloud_test_manifest = os.path.join(args.cloud_data_path, TEST_MANIFEST)
    cloud_test_tar = os.path.join(args.cloud_data_path, TEST_TAR)
    cloud_vocab_file = os.path.join(args.cloud_data_path, VOCAB_FILE)
    cloud_mean_file = os.path.join(args.cloud_data_path, MEAN_STD_FILE)

    local_train_manifest = os.path.join(args.local_tmp_path, TRAIN_MANIFEST)
    local_train_tar = os.path.join(args.local_tmp_path, TRAIN_TAR)
    local_test_manifest = os.path.join(args.local_tmp_path, TEST_MANIFEST)
    local_test_tar = os.path.join(args.local_tmp_path, TEST_TAR)

    if os.path.exists(args.local_tmp_path):
        shutil.rmtree(args.local_tmp_path)
    os.makedirs(args.local_tmp_path)

    # train data
    if args.train_manifest_path != "":
        ret = pcloud_exist(cloud_train_manifest)
        if ret != 0:
            pack_data(args.train_manifest_path, local_train_tar,
                      local_train_manifest)
            pcloud_cp(local_train_manifest, cloud_train_manifest)
            pcloud_cp(local_train_tar, cloud_train_tar)

    # test data
    if args.test_manifest_path != "":
        ret = pcloud_exist(cloud_test_manifest)
        if ret != 0:
            pack_data(args.test_manifest_path, local_test_tar,
                      local_test_manifest)
            pcloud_cp(local_test_manifest, cloud_test_manifest)
            pcloud_cp(local_test_tar, cloud_test_tar)

    # vocab file
    if args.vocab_file != "":
        ret = pcloud_exist(cloud_vocab_file)
        if ret != 0:
            pcloud_cp(args.vocab_file, cloud_vocab_file)

    # mean_std file
    if args.mean_std_file != "":
        ret = pcloud_exist(cloud_mean_file)
        if ret != 0:
            pcloud_cp(args.mean_std_file, cloud_mean_file)

    shutil.rmtree(args.local_tmp_path)
