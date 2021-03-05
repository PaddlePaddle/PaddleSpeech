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

# Released Models

## Language Model Released

Language Model | Training Data | Token-based | Size | Descriptions
:-------------:| :------------:| :-----: | -----: | :-----------------
[English LM](https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm) |  [CommonCrawl(en.00)](http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.00.deduped.xz) | Word-based | 8.3 GB | Pruned with 0 1 1 1 1; <br/> About 1.85 billion n-grams; <br/> 'trie'  binary with '-a 22 -q 8 -b 8'
[Mandarin LM Small](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm) | Baidu Internal Corpus | Char-based | 2.8 GB | Pruned with 0 1 2 4 4; <br/> About 0.13 billion n-grams; <br/> 'probing' binary with default settings
[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) | Baidu Internal Corpus | Char-based | 70.4 GB | No Pruning; <br/> About 3.7 billion n-grams; <br/> 'probing' binary with default settings
