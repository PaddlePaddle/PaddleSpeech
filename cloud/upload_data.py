"""This script is used for preparing data for DeepSpeech2 trainning on paddle
cloud.

Steps:
1. Read original manifest and get the local path of sound files.
2. Tar all local sound files into one tar file.
3. Modify original manifest to remove the local path information.

Finally, we will get a tar file and a new manifest.
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
from subprocess import call
import _init_paths
from data_utils.utils import read_manifest

TRAIN_TAR = "cloud.train.tar"
TRAIN_MANIFEST = "cloud.train.manifest"
DEV_TAR = "cloud.dev.tar"
DEV_MANIFEST = "cloud.dev.manifest"
VOCAB_FILE = "vocab.txt"
MEAN_STD_FILE = "mean_std.npz"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--train_manifest_path",
    default="../datasets/manifest.train",
    type=str,
    help="Manifest file path for train data. (default: %(default)s)")
parser.add_argument(
    "--dev_manifest_path",
    default="../datasets/manifest.dev",
    type=str,
    help="Manifest file path for validation data. (default: %(default)s)")
parser.add_argument(
    "--vocab_file",
    default="../datasets/vocab/eng_vocab.txt",
    type=str,
    help="Vocabulary file to be uploaded to paddlecloud. "
    "(default: %(default)s)")
parser.add_argument(
    "--mean_std_file",
    default="../mean_std.npz",
    type=str,
    help="Normalizer's statistics (mean and stddev) file to be uploaded to "
    "paddlecloud. (default: %(default)s)")
parser.add_argument(
    "--cloud_data_path",
    required=True,
    type=str,
    help="Destination path on paddlecloud. (default: %(default)s)")
parser.add_argument(
    "--local_tmp_path",
    default="./tmp/",
    type=str,
    help="Local directory for storing temporary data. (default: %(default)s)")
args = parser.parse_args()


def pack_data(manifest_path, out_tar_path, out_manifest_path):
    """1. According to the manifest, tar sound files into out_tar_path.
    2. Generate a new manifest for output tar file.
    """
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


def pcloud_mkdir(dir):
    """Make directory in PaddleCloud filesystem.
    """
    if call(['paddlecloud', 'mkdir', dir]) != 0:
        raise IOError("PaddleCloud mkdir failed: %s." % dir)


def pcloud_cp(src, dst):
    """Copy src from local filesytem to dst in PaddleCloud filesystem,
    or downlowd src from PaddleCloud filesystem to dst in local filesystem.
    """
    if call(['paddlecloud', 'cp', src, dst]) != 0:
        raise IOError("PaddleCloud cp failed: from [%s] to [%s]." % (src, dst))


def pcloud_exist(path):
    """Check if file or directory exists in PaddleCloud filesystem.
    """
    ret = call(['paddlecloud', 'ls', path])
    return ret


if __name__ == '__main__':
    cloud_train_manifest = os.path.join(args.cloud_data_path, TRAIN_MANIFEST)
    cloud_train_tar = os.path.join(args.cloud_data_path, TRAIN_TAR)
    cloud_dev_manifest = os.path.join(args.cloud_data_path, DEV_MANIFEST)
    cloud_dev_tar = os.path.join(args.cloud_data_path, DEV_TAR)
    cloud_vocab_file = os.path.join(args.cloud_data_path, VOCAB_FILE)
    cloud_mean_file = os.path.join(args.cloud_data_path, MEAN_STD_FILE)

    local_train_manifest = os.path.join(args.local_tmp_path, TRAIN_MANIFEST)
    local_train_tar = os.path.join(args.local_tmp_path, TRAIN_TAR)
    local_dev_manifest = os.path.join(args.local_tmp_path, DEV_MANIFEST)
    local_dev_tar = os.path.join(args.local_tmp_path, DEV_TAR)

    # prepare local and cloud dir
    if os.path.exists(args.local_tmp_path):
        shutil.rmtree(args.local_tmp_path)
    os.makedirs(args.local_tmp_path)
    pcloud_mkdir(args.cloud_data_path)

    # pack and upload train data
    pack_data(args.train_manifest_path, local_train_tar, local_train_manifest)
    pcloud_cp(local_train_manifest, cloud_train_manifest)
    pcloud_cp(local_train_tar, cloud_train_tar)

    # pack and upload validation data
    pack_data(args.dev_manifest_path, local_dev_tar, local_dev_manifest)
    pcloud_cp(local_dev_manifest, cloud_dev_manifest)
    pcloud_cp(local_dev_tar, cloud_dev_tar)

    # upload vocab file and mean_std file
    pcloud_cp(args.vocab_file, cloud_vocab_file)
    pcloud_cp(args.mean_std_file, cloud_mean_file)

    shutil.rmtree(args.local_tmp_path)
