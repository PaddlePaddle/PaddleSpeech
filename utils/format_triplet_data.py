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
"""format manifest with more metadata."""
import argparse
import functools
import json

import jsonlines

from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.frontend.utility import load_cmvn
from paddlespeech.s2t.io.utility import feat_type
from paddlespeech.s2t.utils.utility import add_arguments
from paddlespeech.s2t.utils.utility import print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('cmvn_path',       str,
        'examples/librispeech/data/mean_std.json',
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
    print_arguments(args, globals())
    fout = open(args.output_path, 'w', encoding='utf-8')

    # get feat dim
    mean, std = load_cmvn(args.cmvn_path, filetype='json')
    feat_dim = mean.shape[0] #(D)
    print(f"Feature dim: {feat_dim}")

    text_feature = TextFeaturizer(args.unit_type, args.vocab_path, args.spm_model_prefix)
    vocab_size = text_feature.vocab_size
    print(f"Vocab size: {vocab_size}")

    count = 0
    for manifest_path in args.manifest_paths:
        with jsonlines.open(str(manifest_path), 'r') as reader:
            manifest_jsons = list(reader)
        for line_json in manifest_jsons:
            # text: translation text, text1: transcript text.
            # Currently only support joint-vocab, will add separate vocabs setting.
            line = line_json['text']
            tokens = text_feature.tokenize(line)
            tokenids = text_feature.featurize(line)
            line_json['token'] = tokens
            line_json['token_id'] = tokenids
            line_json['token_shape'] = (len(tokenids), vocab_size)
            line = line_json['text1']
            tokens = text_feature.tokenize(line)
            tokenids = text_feature.featurize(line)
            line_json['token1'] = tokens
            line_json['token_id1'] = tokenids
            line_json['token_shape1'] = (len(tokenids), vocab_size)

            feat_shape = line_json['feat_shape']
            assert isinstance(feat_shape, (list, tuple)), type(feat_shape)
            filetype = feat_type(line_json['feat'])
            if filetype == 'sound':
                feat_shape.append(feat_dim)
            else: # kaldi
                raise NotImplementedError('no support kaldi feat now!')
            fout.write(json.dumps(line_json) + '\n')
            count += 1

    print(f"Examples number: {count}")
    fout.close()


if __name__ == '__main__':
    main()
