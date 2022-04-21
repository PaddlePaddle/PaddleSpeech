# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

pretrained_models = {
    # The tags for pretrained_models should be "{model_name}[_{dataset}][-{lang}][-...]".
    # e.g. "conformer_wenetspeech-zh-16k", "transformer_aishell-zh-16k" and "panns_cnn6-32k".
    # Command line and python api use "{model_name}[_{dataset}]" as --model, usage:
    # "paddlespeech asr --model conformer_wenetspeech --lang zh --sr 16000 --input ./input.wav"
    "ernie_linear_p7_wudao-punc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/text/ernie_linear_p7_wudao-punc-zh.tar.gz',
        'md5':
        '12283e2ddde1797c5d1e57036b512746',
        'cfg_path':
        'ckpt/model_config.json',
        'ckpt_path':
        'ckpt/model_state.pdparams',
        'vocab_file':
        'punc_vocab.txt',
    },
    "ernie_linear_p3_wudao-punc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/text/ernie_linear_p3_wudao-punc-zh.tar.gz',
        'md5':
        '448eb2fdf85b6a997e7e652e80c51dd2',
        'cfg_path':
        'ckpt/model_config.json',
        'ckpt_path':
        'ckpt/model_state.pdparams',
        'vocab_file':
        'punc_vocab.txt',
    },
}

model_alias = {
    "ernie_linear_p7": "paddlespeech.text.models:ErnieLinear",
    "ernie_linear_p3": "paddlespeech.text.models:ErnieLinear",
}

tokenizer_alias = {
    "ernie_linear_p7": "paddlenlp.transformers:ErnieTokenizer",
    "ernie_linear_p3": "paddlenlp.transformers:ErnieTokenizer",
}
