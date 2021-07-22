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
import argparse

from text_processing import normalization

parser = argparse.ArgumentParser(
    description="Normalize text in Chinese with some rules.")
parser.add_argument("input", type=str, help="the input sentences")
parser.add_argument("output", type=str, help="path to save the output file.")
args = parser.parse_args()

with open(args.input, 'rt') as fin:
    with open(args.output, 'wt') as fout:
        for sent in fin:
            sent = normalization.normalize_sentence(sent.strip())
            fout.write(sent)
            fout.write('\n')
