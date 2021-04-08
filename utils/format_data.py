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
"""format manifest with more metadata."""
import argparse
import functools
import json
from collections import Counter
import os
import copy
import tempfile

from deepspeech.frontend.utility import read_manifest
from deepspeech.frontend.utility import UNK
from deepspeech.frontend.utility import BLANK
from deepspeech.frontend.utility import SOS
from deepspeech.utils.utility import add_arguments
from deepspeech.utils.utility import print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('feat_type', str, "raw", "speech feature type, e.g. raw(wav, flac), kaldi")
add_arg('unit_type', str, "character", "Unit type, e.g. character, word, bpe")
add_arg('vocab_path',       str,
        'examples/librispeech/data/vocab.txt',
        "Filepath to write the vocabulary.")
add_arg('manifest_paths',   str,
        None,
        "Filepaths of manifests for building vocabulary. "
        "You can provide multiple manifest files.",
        nargs='+',
        required=True)
# bpe
add_arg('bpe_model_prefix', str, "bpe_model_%(bpe_mode)_%(count_threshold)", "bpe model prefix, only need when `unit_type` is bpe")
add_arg('output_path',  str, None, "filepath of formated manifest.", required=True)
# yapf: disable
args = parser.parse_args()


def main():
    print_arguments(args)

    # read vocab
    vocab = dict()
    with open(args.vocab_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            token = line.strip()
            vocab[token] = len(vocab)
    vocab_size = len(vocab)

    fout = open(args.output_path, 'w', encoding='utf-8')

    if args.unit_type != 'bpe':
        for manifest_path in args.manifest_paths:
            manifest_jsons = read_manifest(manifest_path)
            for line_json in manifest_jsons:
                tokens = []
                tokenids = []
                if args.unit_type == 'character':
                    for char in line_json['text']:
                        tokens.append(char)
                        tokenids.append(vocab[char])
                elif args.unit_type == 'word':
                    for word in line_json['text'].split():
                        tokens.append(word)
                        tokenids.append(vocab[word])
                line_json['token'] = tokens
                line_json['token_id'] = tokenids
                line_json['token_shape'] = (len(tokenids), vocab_size)
                fout.write(json.dumps(line_json) + '\n')
    else:
        import sentencepiece as spm

        # encode
        sp = spm.SentencePieceProcessor()
        sp.Load(args.bpe_model_prefix + '.model')

        def valid(line):
            return True

        def encode(l):
            return sp.EncodeAsPieces(l)

        def encode_line(line):
            line = line.strip()
            if len(line) > 0:
                line = encode(line)
                if valid(line):
                    return line
                else:
                    stats["num_filtered"] += 1
            else:
                stats["num_empty"] += 1
            return None

        for manifest_path in args.manifest_paths:
            manifest_jsons = read_manifest(manifest_path)
            for line_json in manifest_jsons:
                line = line_json['text']
                tokens = []
                tokenids = []
                enc_line = encode_line(line)
                for code in enc_line:
                    tokens.append(code)
                    tokenids.append(vocab[code])
                    #print(code, vocab[code])
                line_json['token'] = tokens
                line_json['token_id'] = tokenids
                line_json['token_shape'] = (len(tokenids), vocab_size)
                fout.write(json.dumps(line_json) + '\n')

    fout.close()


if __name__ == '__main__':
    main()
