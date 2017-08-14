"""This tool is used for splitting data into each node of
paddle cloud by total trainer count and current trainer id.
The meaning of trainer is a instance of k8s cluster.
This script should be called in paddle cloud.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--in_manifest_path",
    default='./cloud.train.manifest',
    type=str,
    help="Input manifest path. (default: %(default)s)")
parser.add_argument(
    "--data_tar_path",
    default='./cloud.train.tar',
    type=str,
    help="Data tar file path. (default: %(default)s)")
parser.add_argument(
    "--out_manifest_path",
    default='./local.train.manifest',
    type=str,
    help="Out manifest file path. (default: %(default)s)")
args = parser.parse_args()


def split_data(in_manifest, tar_path, out_manifest):
    with open("/trainer_id", "r") as f:
        trainer_id = int(f.readline()[:-1])
    with open("/trainer_count", "r") as f:
        trainer_count = int(f.readline()[:-1])

    tar_path = os.path.abspath(tar_path)
    result = []
    for index, json_line in enumerate(open(in_manifest)):
        if (index % trainer_count) == trainer_id:
            json_data = json.loads(json_line)
            json_data['audio_filepath'] = "tar:%s#%s" % (
                tar_path, json_data['audio_filepath'])
            result.append("%s\n" % json.dumps(json_data))
    with open(out_manifest, 'w') as manifest:
        manifest.writelines(result)


if __name__ == '__main__':
    split_data(args.in_manifest_path, args.data_tar_path,
               args.out_manifest_path)
