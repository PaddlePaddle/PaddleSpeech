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
"""Build vocabulary from manifest files.

Each item in vocabulary file is a character.
"""

import argparse
import functools
import codecs
import json
from collections import Counter
import os.path
import _init_paths
from data_utils.utility import read_manifest
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('count_threshold',  int,    0,  "Truncation threshold for char counts.")
add_arg('vocab_path',       str,
        'data/librispeech/vocab.txt',
        "Filepath to write the vocabulary.")
add_arg('manifest_paths',   str,
        None,
        "Filepaths of manifests for building vocabulary. "
        "You can provide multiple manifest files.",
        nargs='+',
        required=True)
# yapf: disable
args = parser.parse_args()


def count_manifest(counter, manifest_path):
    manifest_jsons = read_manifest(manifest_path)
    for line_json in manifest_jsons:
        for char in line_json['text']:
            counter.update(char)


def main():
    print_arguments(args)

    counter = Counter()
    for manifest_path in args.manifest_paths:
        count_manifest(counter, manifest_path)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with codecs.open(args.vocab_path, 'w', 'utf-8') as fout:
        fout.write('<unk>' + '\n')
        for char, count in count_sorted:
            if count < args.count_threshold: break
            fout.write(char + '\n')


if __name__ == '__main__':
    main()
