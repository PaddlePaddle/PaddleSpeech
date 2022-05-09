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
    "deepspeech2online_aishell-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_fbank161_ckpt_0.2.1.model.tar.gz',
        'md5':
        '98b87b171b7240b7cae6e07d8d0bc9be',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/deepspeech2_online/checkpoints/avg_1',
        'model':
        'exp/deepspeech2_online/checkpoints/avg_1.jit.pdmodel',
        'params':
        'exp/deepspeech2_online/checkpoints/avg_1.jit.pdiparams',
        'lm_url':
        'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
        'lm_md5':
        '29e02312deb2e59b3c8686c7966d4fe3'
    },
    "conformer_online_multicn-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/multi_cn/asr1/asr1_chunk_conformer_multi_cn_ckpt_0.2.3.model.tar.gz',
        'md5':
        '0ac93d390552336f2a906aec9e33c5fa',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/chunk_conformer/checkpoints/multi_cn',
        'model':
        'exp/chunk_conformer/checkpoints/multi_cn.pdparams',
        'params':
        'exp/chunk_conformer/checkpoints/multi_cn.pdparams',
        'lm_url':
        'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
        'lm_md5':
        '29e02312deb2e59b3c8686c7966d4fe3'
    },
    "conformer_online_wenetspeech-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model.tar.gz',
        'md5':
        'b8c02632b04da34aca88459835be54a6',
        'cfg_path':
        'model.yaml',
        'ckpt_path':
        'exp/chunk_conformer/checkpoints/avg_10',
        'model':
        'exp/chunk_conformer/checkpoints/avg_10.pdparams',
        'params':
        'exp/chunk_conformer/checkpoints/avg_10.pdparams',
        'lm_url':
        '',
        'lm_md5':
        '',
    },
}
