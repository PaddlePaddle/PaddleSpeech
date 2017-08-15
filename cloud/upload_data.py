"""This script is for uploading data for DeepSpeech2 training on paddlecloud.

Steps:
1. Read original manifests and extract local sound files.
2. Tar all local sound files into multiple tar files and upload them.
3. Modify original manifests with updated paths in cloud filesystem.
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--in_manifest_paths",
    default=["../datasets/manifest.test", "../datasets/manifest.dev"],
    type=str,
    nargs='+',
    help="Local filepaths of input manifests to load, pack and upload."
    "(default: %(default)s)")
parser.add_argument(
    "--out_manifest_paths",
    default=["./cloud.manifest.test", "./cloud.manifest.dev"],
    type=str,
    nargs='+',
    help="Local filepaths of modified manifests to write to. "
    "(default: %(default)s)")
parser.add_argument(
    "--cloud_data_dir",
    required=True,
    type=str,
    help="Destination directory on paddlecloud to upload data to.")
parser.add_argument(
    "--num_shards",
    default=10,
    type=int,
    help="Number of parts to split data to. (default: %(default)s)")
parser.add_argument(
    "--local_tmp_dir",
    default="./tmp/",
    type=str,
    help="Local directory for storing temporary data. (default: %(default)s)")
args = parser.parse_args()


def upload_data(in_manifest_path_list, out_manifest_path_list, local_tmp_dir,
                upload_tar_dir, num_shards):
    """Extract and pack sound files listed in the manifest files into multple
    tar files and upload them to padldecloud. Besides, generate new manifest
    files with updated paths in paddlecloud.
    """
    # compute total audio number
    total_line = 0
    for manifest_path in in_manifest_path_list:
        with open(manifest_path, 'r') as f:
            total_line += len(f.readlines())
    line_per_tar = (total_line // num_shards) + 1

    # pack and upload shard by shard
    line_count, tar_file = 0, None
    for manifest_path, out_manifest_path in zip(in_manifest_path_list,
                                                out_manifest_path_list):
        manifest = read_manifest(manifest_path)
        out_manifest = []
        for json_data in manifest:
            sound_filepath = json_data['audio_filepath']
            sound_filename = os.path.basename(sound_filepath)
            if line_count % line_per_tar == 0:
                if tar_file != None:
                    tar_file.close()
                    pcloud_cp(tar_path, upload_tar_dir)
                    os.remove(tar_path)
                tar_name = 'part-%s-of-%s.tar' % (
                    str(line_count // line_per_tar).zfill(5),
                    str(num_shards).zfill(5))
                tar_path = os.path.join(local_tmp_dir, tar_name)
                tar_file = tarfile.open(tar_path, 'w')
            tar_file.add(sound_filepath, arcname=sound_filename)
            line_count += 1
            json_data['audio_filepath'] = "tar:%s#%s" % (
                os.path.join(upload_tar_dir, tar_name), sound_filename)
            out_manifest.append("%s\n" % json.dumps(json_data))
        with open(out_manifest_path, 'w') as f:
            f.writelines(out_manifest)
    tar_file.close()
    pcloud_cp(tar_path, upload_tar_dir)
    os.remove(tar_path)


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


if __name__ == '__main__':
    if not os.path.exists(args.local_tmp_dir):
        os.makedirs(args.local_tmp_dir)
    pcloud_mkdir(args.cloud_data_dir)

    upload_data(args.in_manifest_paths, args.out_manifest_paths,
                args.local_tmp_dir, args.cloud_data_dir, 10)

    shutil.rmtree(args.local_tmp_dir)
