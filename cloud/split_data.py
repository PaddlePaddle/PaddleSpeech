"""This tool is used for splitting data into each node of
paddlecloud. This script should be called in paddlecloud.
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
    type=str,
    required=True,
    help="Input manifest path for all nodes.")
parser.add_argument(
    "--out_manifest_path",
    type=str,
    required=True,
    help="Output manifest file path for current node.")
args = parser.parse_args()


def split_data(in_manifest_path, out_manifest_path):
    with open("/trainer_id", "r") as f:
        trainer_id = int(f.readline()[:-1])
    with open("/trainer_count", "r") as f:
        trainer_count = int(f.readline()[:-1])

    out_manifest = []
    for index, json_line in enumerate(open(in_manifest_path, 'r')):
        if (index % trainer_count) == trainer_id:
            out_manifest.append("%s\n" % json_line.strip())
    with open(out_manifest_path, 'w') as f:
        f.writelines(out_manifest)


if __name__ == '__main__':
    split_data(args.in_manifest_path, args.out_manifest_path)
