#!/usr/bin/env python3
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
import os
import tempfile
from collections import Counter

import jsonlines

from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.frontend.utility import BLANK
from paddlespeech.s2t.frontend.utility import SOS
from paddlespeech.s2t.frontend.utility import SPACE
from paddlespeech.s2t.frontend.utility import UNK
from paddlespeech.s2t.utils.utility import add_arguments
from paddlespeech.s2t.utils.utility import print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('unit_type', str, "char", "Unit type, e.g. char, word, spm")
add_arg('count_threshold', int, 0,
        "Truncation threshold for char/word counts.Default 0, no truncate.")
add_arg('vocab_path', str,
        'examples/librispeech/data/vocab.txt',
        "Filepath to write the vocabulary.")
add_arg('manifest_paths', str,
        None,
        "Filepaths of manifests for building vocabulary. "
        "You can provide multiple manifest files.",
        nargs='+',
        required=True)
add_arg('text_keys', str,
        'text',
        "keys of the text in manifest for building vocabulary. "
        "You can provide multiple k.",
        nargs='+')
# bpe
add_arg('spm_vocab_size', int, 0, "Vocab size for spm.")
add_arg('spm_mode', str, 'unigram', "spm model type, e.g. unigram, spm, char, word. only need when `unit_type` is spm")
add_arg('spm_model_prefix', str, "", "spm_model_%(spm_mode)_%(count_threshold), spm model prefix, only need when `unit_type` is spm")
add_arg('spm_character_coverage', float, 0.9995, "character coverage to determine the minimum symbols")

# yapf: disable
args = parser.parse_args()


def count_manifest(counter, text_feature, manifest_path):
    manifest_jsons = []
    with jsonlines.open(manifest_path, 'r') as reader:
        for json_data in reader:
            manifest_jsons.append(json_data)

    for line_json in manifest_jsons:
        if isinstance(line_json['text'], str):
            line = text_feature.tokenize(line_json['text'], replace_space=False)
            counter.update(line)
        else:
            assert isinstance(line_json['text'], list)
            for text in line_json['text']:
                line = text_feature.tokenize(text, replace_space=False)
                counter.update(line)

def dump_text_manifest(fileobj, manifest_path, key='text'):
    manifest_jsons = []
    with jsonlines.open(manifest_path, 'r') as reader:
        for json_data in reader:
            manifest_jsons.append(json_data)

    for line_json in manifest_jsons:
        if isinstance(line_json[key], str):
            fileobj.write(line_json[key] + "\n")
        else:
            assert isinstance(line_json[key], list)
            for line in line_json[key]:
                fileobj.write(line + "\n")

def main():
    print_arguments(args, globals())

    fout = open(args.vocab_path, 'w', encoding='utf-8')
    fout.write(BLANK + "\n")  # 0 will be used for "blank" in CTC
    fout.write(UNK + '\n')  # <unk> must be 1

    if args.unit_type == 'spm':
        # tools/spm_train --input=$wave_data/lang_char/input.txt
        # --vocab_size=${nbpe} --model_type=${bpemode}
        # --model_prefix=${bpemodel} --input_sentence_size=100000000
        import sentencepiece as spm

        fp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        for manifest_path in args.manifest_paths:
            text_keys = [args.text_keys] if type(args.text_keys) is not list else args.text_keys
            for text_key in text_keys:
                dump_text_manifest(fp, manifest_path, key=text_key)
        fp.close()
        # train
        spm.SentencePieceTrainer.Train(
            input=fp.name,
            vocab_size=args.spm_vocab_size,
            model_type=args.spm_mode,
            model_prefix=args.spm_model_prefix,
            input_sentence_size=100000000,
            character_coverage=args.spm_character_coverage)
        os.unlink(fp.name)

    # encode
    text_feature = TextFeaturizer(args.unit_type, "", args.spm_model_prefix)
    counter = Counter()

    for manifest_path in args.manifest_paths:
        count_manifest(counter, text_feature, manifest_path)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    tokens = []
    for token, count in count_sorted:
        if count < args.count_threshold:
            break
        # replace space by `<space>`
        token = SPACE if token == ' ' else token
        tokens.append(token)

    tokens = sorted(tokens)
    for token in tokens:
        fout.write(token + '\n')

    fout.write(SOS + "\n")  # <sos/eos>
    fout.close()


if __name__ == '__main__':
    main()
