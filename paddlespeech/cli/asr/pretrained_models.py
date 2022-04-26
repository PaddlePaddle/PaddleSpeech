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
    # e.g. "conformer_wenetspeech-zh-16k" and "panns_cnn6-32k".
    # Command line and python api use "{model_name}[_{dataset}]" as --model, usage:
    # "paddlespeech asr --model conformer_wenetspeech --lang zh --sr 16000 --input ./input.wav"
    "conformer_wenetspeech-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1_conformer_wenetspeech_ckpt_0.1.1.model.tar.gz',
        'md5':
        '76cb19ed857e6623856b7cd7ebbfeda4',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/conformer/checkpoints/wenetspeech',
    },
    "conformer_aishell-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr1/asr1_conformer_aishell_ckpt_0.1.2.model.tar.gz',
        'md5':
        '3f073eccfa7bb14e0c6867d65fc0dc3a',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/conformer/checkpoints/avg_30',
    },
    "conformer_online_aishell-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr1/asr1_chunk_conformer_aishell_ckpt_0.2.0.model.tar.gz',
        'md5':
        'b374cfb93537761270b6224fb0bfc26a',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/chunk_conformer/checkpoints/avg_30',
    },
    "transformer_librispeech-en-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr1/asr1_transformer_librispeech_ckpt_0.1.1.model.tar.gz',
        'md5':
        '2c667da24922aad391eacafe37bc1660',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/transformer/checkpoints/avg_10',
    },
    "deepspeech2offline_aishell-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz',
        'md5':
        '932c3593d62fe5c741b59b31318aa314',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/deepspeech2/checkpoints/avg_1',
        'lm_url':
        'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
        'lm_md5':
        '29e02312deb2e59b3c8686c7966d4fe3'
    },
    "deepspeech2online_aishell-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_fbank161_ckpt_0.2.1.model.tar.gz',
        'md5':
        '98b87b171b7240b7cae6e07d8d0bc9be',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/deepspeech2_online/checkpoints/avg_1',
        'lm_url':
        'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
        'lm_md5':
        '29e02312deb2e59b3c8686c7966d4fe3'
    },
    "deepspeech2offline_librispeech-en-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr0/asr0_deepspeech2_librispeech_ckpt_0.1.1.model.tar.gz',
        'md5':
        'f5666c81ad015c8de03aac2bc92e5762',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/deepspeech2/checkpoints/avg_1',
        'lm_url':
        'https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm',
        'lm_md5':
        '099a601759d467cd0a8523ff939819c5'
    },
}

model_alias = {
    "deepspeech2offline":
    "paddlespeech.s2t.models.ds2:DeepSpeech2Model",
    "deepspeech2online":
    "paddlespeech.s2t.models.ds2_online:DeepSpeech2ModelOnline",
    "conformer":
    "paddlespeech.s2t.models.u2:U2Model",
    "conformer_online":
    "paddlespeech.s2t.models.u2:U2Model",
    "transformer":
    "paddlespeech.s2t.models.u2:U2Model",
    "wenetspeech":
    "paddlespeech.s2t.models.u2:U2Model",
}
