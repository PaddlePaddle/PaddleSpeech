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


def add_arg(argname, type, default, help, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    parser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


# yapf: disable
parser = argparse.ArgumentParser(description=__doc__)
add_arg('count_threshold',  int,    0,  "Truncation threshold for char counts.")
add_arg('vocab_path',       str,
        'datasets/vocab/zh_vocab.txt',
        "Filepath to write the vocabulary.")
add_arg('manifest_paths',   str,
        None,
        "Filepaths of manifests for building vocabulary. "
        "You can provide multiple manifest files.",
        nargs='+',
        required=True)
args = parser.parse_args()
# yapf: disable


def count_manifest(counter, manifest_path):
    manifest_jsons = utils.read_manifest(manifest_path)
    for line_json in manifest_jsons:
        for char in line_json['text']:
            counter.update(char)


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).iteritems()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def main():
    print_arguments(args)

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
