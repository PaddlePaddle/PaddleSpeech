"""Build vocabulary dictionary from manifest files.

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

parser = argparse.ArgumentParser(
    description='Build vocabulary dictionary from transcription texts.')
parser.add_argument(
    "--manifest_paths",
    type=str,
    help="Manifest paths for building vocabulary dictionary."
    "You can provide multiple manifest files.",
    nargs='+',
    required=True)
parser.add_argument(
    "--count_threshold",
    default=0,
    type=int,
    help="Characters whose count below the threshold will be truncated. "
    "(default: %(default)s)")
parser.add_argument(
    "--vocab_path",
    default='datasets/vocab/zh_vocab.txt',
    type=str,
    help="Filepath to write vocabularies. (default: %(default)s)")
args = parser.parse_args()


def count_manifest(counter, manifest_path):
    for json_line in codecs.open(manifest_path, 'r', 'utf-8'):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise Exception('Error parsing manifest: %s, %s' % \
                    (manifest_path, e))
        text = json_data['text']
        for char in text:
            counter.update(char)


def main():
    counter = Counter()
    for manifest_path in args.manifest_paths:
        count_manifest(counter, manifest_path)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with codecs.open(args.vocab_path, 'w', 'utf-8') as fout:
        for item_pair in count_sorted:
            if item_pair[1] < args.count_threshold: break
            fout.write(item_pair[0] + '\n')


if __name__ == '__main__':
    main()
