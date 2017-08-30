"""Build vocabulary from manifest files.

Each item in vocabulary file is a character.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import json
from collections import Counter
import os.path
import _init_paths
from data_utils import utils

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--manifest_paths",
    type=str,
    help="Manifest paths for building vocabulary."
    "You can provide multiple manifest files.",
    nargs='+',
    required=True)
parser.add_argument(
    "--count_threshold",
    default=0,
    type=int,
    help="Characters whose counts are below the threshold will be truncated. "
    "(default: %(default)i)")
parser.add_argument(
    "--vocab_path",
    default='datasets/vocab/zh_vocab.txt',
    type=str,
    help="File path to write the vocabulary. (default: %(default)s)")
args = parser.parse_args()


def count_manifest(counter, manifest_path):
    manifest_jsons = utils.read_manifest(manifest_path)
    for line_json in manifest_jsons:
        for char in line_json['text']:
            counter.update(char)


def main():
    counter = Counter()
    for manifest_path in args.manifest_paths:
        count_manifest(counter, manifest_path)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with codecs.open(args.vocab_path, 'w', 'utf-8') as fout:
        for char, count in count_sorted:
            if count < args.count_threshold: break
            fout.write(char + '\n')


if __name__ == '__main__':
    main()
