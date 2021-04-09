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
from deepspeech.frontend.utility import load_cmvn
from deepspeech.utils.utility import add_arguments
from deepspeech.utils.utility import print_arguments
from deepspeech.frontend.featurizer.text_featurizer import TextFeaturizer

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('feat_type', str, "raw", "speech feature type, e.g. raw(wav, flac), kaldi")
add_arg('cmvn_path',       str,
        'examples/librispeech/data/mean_std.npz',
        "Filepath of cmvn.")
add_arg('unit_type', str, "char", "Unit type, e.g. char, word, spm")
add_arg('vocab_path',       str,
        'examples/librispeech/data/vocab.txt',
        "Filepath of the vocabulary.")
add_arg('manifest_paths',   str,
        None,
        "Filepaths of manifests for building vocabulary. "
        "You can provide multiple manifest files.",
        nargs='+',
        required=True)
# bpe
add_arg('spm_model_prefix', str, None,
     "spm model prefix, spm_model_%(bpe_mode)_%(count_threshold), only need when `unit_type` is spm")
add_arg('output_path',  str, None, "filepath of formated manifest.", required=True)
# yapf: disable
args = parser.parse_args()


def main():
    print_arguments(args)
    fout = open(args.output_path, 'w', encoding='utf-8')

    # get feat dim
    mean, std = load_cmvn(args.cmvn_path, filetype='npz')
    feat_dim = mean.shape[0]
    print(f"Feature dim: {feat_dim}")

    text_feature = TextFeaturizer(args.unit_type, args.vocab_path, args.spm_model_prefix)
    vocab_size = text_feature.vocab_size
    print(f"Vocab size: {vocab_size}")

    count = 0
    for manifest_path in args.manifest_paths:
        manifest_jsons = read_manifest(manifest_path)
        for line_json in manifest_jsons:
            line = line_json['text']
            tokens = text_feature.tokenize(line)
            tokenids = text_feature.featurize(line)
            line_json['token'] = tokens
            line_json['token_id'] = tokenids
            line_json['token_shape'] = (len(tokenids), vocab_size)
            feat_shape = line_json['feat_shape']
            assert isinstance(feat_shape, (list, tuple)), type(feat_shape)
            if args.feat_type == 'raw':
                feat_shape.append(feat_dim)
            else: # kaldi
                raise NotImplemented('no support kaldi feat now!')
            fout.write(json.dumps(line_json) + '\n')
            count += 1
            
    print(f"Examples number: {count}")
    fout.close()


if __name__ == '__main__':
    main()
