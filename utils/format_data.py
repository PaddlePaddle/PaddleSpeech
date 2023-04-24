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
    filetype = args.cmvn_path.split(".")[-1]
    mean, istd = load_cmvn(args.cmvn_path, filetype=filetype)
    feat_dim = mean.shape[0] #(D)
    print(f"Feature dim: {feat_dim}")

    text_feature = TextFeaturizer(args.unit_type, args.vocab_path, args.spm_model_prefix)
    vocab_size = text_feature.vocab_size
    print(f"Vocab size: {vocab_size}")

    # josnline like this
    # {
    #   "input": [{"name": "input1", "shape": (100, 83), "feat": "xxx.ark:123"}],
    #   "output": [{"name":"target1", "shape": (40, 5002), "text": "a b c de"}],
    #   "utt2spk": "111-2222",
    #   "utt": "111-2222-333"
    # }
    count = 0
    for manifest_path in args.manifest_paths:
        with jsonlines.open(str(manifest_path), 'r') as reader:
            manifest_jsons = list(reader)

        for line_json in manifest_jsons:
            output_json = {
                "input": [],
                "output": [],
                'utt': line_json['utt'],
                'utt2spk': line_json.get('utt2spk', 'global'),
            }

            # output
            line = line_json['text']
            if isinstance(line, str):
                # only one target
                tokens = text_feature.tokenize(line)
                tokenids = text_feature.featurize(line)
                output_json['output'].append({
                    'name': 'target1',
                    'shape': (len(tokenids), vocab_size),
                    'text': line,
                    'token': ' '.join(tokens),
                    'tokenid': ' '.join(map(str, tokenids)),
                })
            else:
                # isinstance(line, list), multi target in one vocab
                for i, item in enumerate(line, 1):
                    tokens = text_feature.tokenize(item)
                    tokenids = text_feature.featurize(item)
                    output_json['output'].append({
                        'name': f'target{i}',
                        'shape': (len(tokenids), vocab_size),
                        'text': item,
                        'token': ' '.join(tokens),
                        'tokenid': ' '.join(map(str, tokenids)),
                    })

            # input
            line = line_json['feat']
            if isinstance(line, str):
                # only one input
                feat_shape = line_json['feat_shape']
                assert isinstance(feat_shape, (list, tuple)), type(feat_shape)
                filetype = feat_type(line)
                if filetype == 'sound':
                    feat_shape.append(feat_dim)
                else: # kaldi
                    raise NotImplementedError('no support kaldi feat now!')

                output_json['input'].append({
                    "name": "input1",
                    "shape": feat_shape,
                    "feat": line,
                    "filetype": filetype,
                })
            else:
                # isinstance(line, list), multi input 
                raise NotImplementedError("not support multi input now!")

            fout.write(json.dumps(output_json) + '\n')
            count += 1

    print(f"{args.manifest_paths} Examples number: {count}")
    fout.close()


if __name__ == '__main__':
    main()
