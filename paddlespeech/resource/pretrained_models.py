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

__all__ = [
    'asr_dynamic_pretrained_models',
    'asr_static_pretrained_models',
    'asr_onnx_pretrained_models',
    'cls_dynamic_pretrained_models',
    'cls_static_pretrained_models',
    'st_dynamic_pretrained_models',
    'st_kaldi_bins',
    'text_dynamic_pretrained_models',
    'tts_dynamic_pretrained_models',
    'tts_static_pretrained_models',
    'tts_onnx_pretrained_models',
    'vector_dynamic_pretrained_models',
    'ssl_dynamic_pretrained_models',
    'whisper_dynamic_pretrained_models',
]

# The tags for pretrained_models should be "{model_name}[_{dataset}][-{lang}][-...]".
# Add code-switch and multilingual tag, "{model_name}[_{dataset}]-[codeswitch/multilingual][_{lang}][-...]".
# e.g. "conformer_wenetspeech-zh-16k" and "panns_cnn6-32k".
# Command line and python api use "{model_name}[_{dataset}]" as --model, usage:
# "paddlespeech asr --model conformer_wenetspeech --lang zh --sr 16000 --input ./input.wav"

# ---------------------------------
# -------------- SSL --------------
# ---------------------------------
ssl_dynamic_pretrained_models = {
    "wav2vec2-en-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr3/wav2vec2-large-960h-lv60-self_ckpt_1.3.0.model.tar.gz',
            'md5':
            'acc46900680e341e500437aa59193518',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'wav2vec2-large-960h-lv60-self',
            'model':
            'wav2vec2-large-960h-lv60-self.pdparams',
            'params':
            'wav2vec2-large-960h-lv60-self.pdparams',
        },
    },
    "wav2vec2ASR_librispeech-en-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr3/wav2vec2ASR-large-960h-librispeech_ckpt_1.3.1.model.tar.gz',
            'md5':
            'cbe28d6c78f3dd2e189968402381f454',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/wav2vec2ASR/checkpoints/avg_1',
            'model':
            'exp/wav2vec2ASR/checkpoints/avg_1.pdparams',
            'params':
            'exp/wav2vec2ASR/checkpoints/avg_1.pdparams',
        },
    },
    "wav2vec2-zh-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr3/wav2vec2-large-wenetspeech-self_ckpt_1.3.0.model.tar.gz',
            'md5':
            '00ea4975c05d1bb58181205674052fe1',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'chinese-wav2vec2-large',
            'model':
            'chinese-wav2vec2-large.pdparams',
            'params':
            'chinese-wav2vec2-large.pdparams',
        },
    },
    "wav2vec2ASR_aishell1-zh-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr3/wav2vec2ASR-large-aishell1_ckpt_1.3.0.model.tar.gz',
            'md5':
            'ac8fa0a6345e6a7535f6fabb5e59e218',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/wav2vec2ASR/checkpoints/avg_1',
            'model':
            'exp/wav2vec2ASR/checkpoints/avg_1.pdparams',
            'params':
            'exp/wav2vec2ASR/checkpoints/avg_1.pdparams',
        },
    },
}

# ---------------------------------
# -------------- ASR --------------
# ---------------------------------
asr_dynamic_pretrained_models = {
    "conformer_wenetspeech-zh-16k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1_conformer_wenetspeech_ckpt_0.1.1.model.tar.gz',
            'md5':
            '76cb19ed857e6623856b7cd7ebbfeda4',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/conformer/checkpoints/wenetspeech',
        },
    },
    "conformer_online_wenetspeech-zh-16k": {
        '1.0': {
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
    },
    "conformer_u2pp_online_wenetspeech-zh-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/asr1_chunk_conformer_u2pp_wenetspeech_ckpt_1.3.0.model.tar.gz',
            'md5':
            '62d230c1bf27731192aa9d3b8deca300',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/chunk_conformer_u2pp/checkpoints/avg_10',
            'model':
            'exp/chunk_conformer_u2pp/checkpoints/avg_10.pdparams',
            'params':
            'exp/chunk_conformer_u2pp/checkpoints/avg_10.pdparams',
            'lm_url':
            '',
            'lm_md5':
            '',
        },
    },
    "conformer_online_multicn-zh-16k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/multi_cn/asr1/asr1_chunk_conformer_multi_cn_ckpt_0.2.0.model.tar.gz',
            'md5':
            '7989b3248c898070904cf042fd656003',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/chunk_conformer/checkpoints/multi_cn',
        },
        '2.0': {
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
            '29e02312deb2e59b3c8686c7966d4fe3',
        },
    },
    "conformer_aishell-zh-16k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr1/asr1_conformer_aishell_ckpt_0.1.2.model.tar.gz',
            'md5':
            '3f073eccfa7bb14e0c6867d65fc0dc3a',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/conformer/checkpoints/avg_30',
        },
    },
    "conformer_online_aishell-zh-16k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr1/asr1_chunk_conformer_aishell_ckpt_0.2.0.model.tar.gz',
            'md5':
            'b374cfb93537761270b6224fb0bfc26a',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/chunk_conformer/checkpoints/avg_30',
        },
    },
    "transformer_librispeech-en-16k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr1/asr1_transformer_librispeech_ckpt_0.1.1.model.tar.gz',
            'md5':
            '2c667da24922aad391eacafe37bc1660',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/transformer/checkpoints/avg_10',
        },
    },
    "deepspeech2online_wenetspeech-zh-16k": {
        '1.0.3': {
            'url':
            'http://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr0/asr0_deepspeech2_online_wenetspeech_ckpt_1.0.3.model.tar.gz',
            'md5':
            'cfe273793e68f790f742b411c98bc75e',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2_online/checkpoints/avg_10',
            'model':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdmodel',
            'params':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdiparams',
            'onnx_model':
            'onnx/model.onnx',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
            'lm_md5':
            '29e02312deb2e59b3c8686c7966d4fe3'
        },
        '1.0.4': {
            'url':
            'http://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr0/asr0_deepspeech2_online_wenetspeech_ckpt_1.0.4.model.tar.gz',
            'md5':
            'c595cb76902b5a5d01409171375989f4',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2_online/checkpoints/avg_10',
            'model':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdmodel',
            'params':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdiparams',
            'onnx_model':
            'onnx/model.onnx',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
            'lm_md5':
            '29e02312deb2e59b3c8686c7966d4fe3'
        },
    },
    "deepspeech2offline_aishell-zh-16k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_offline_aishell_ckpt_1.0.1.model.tar.gz',
            'md5':
            '4d26066c6f19f52087425dc722ae5b13',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2/checkpoints/avg_10',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
            'lm_md5':
            '29e02312deb2e59b3c8686c7966d4fe3'
        },
    },
    "deepspeech2online_aishell-zh-16k": {
        '1.0.2': {
            'url':
            'http://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_fbank161_ckpt_1.0.2.model.tar.gz',
            'md5':
            '4dd42cfce9aaa54db0ec698da6c48ec5',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2_online/checkpoints/avg_1',
            'model':
            'exp/deepspeech2_online/checkpoints/avg_1.jit.pdmodel',
            'params':
            'exp/deepspeech2_online/checkpoints/avg_1.jit.pdiparams',
            'onnx_model':
            'onnx/model.onnx',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
            'lm_md5':
            '29e02312deb2e59b3c8686c7966d4fe3'
        },
    },
    "deepspeech2offline_librispeech-en-16k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/librispeech/asr0/asr0_deepspeech2_offline_librispeech_ckpt_1.0.1.model.tar.gz',
            'md5':
            'ed9e2b008a65268b3484020281ab048c',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2/checkpoints/avg_5',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm',
            'lm_md5':
            '099a601759d467cd0a8523ff939819c5'
        },
    },
    "conformer_talcs-codeswitch_zh_en-16k": {
        '1.4': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/tal_cs/asr1/asr1_conformer_talcs_ckpt_1.4.0.model.tar.gz',
            'md5':
            '01962c5d0a70878fe41cacd4f61e14d1',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/conformer/checkpoints/avg_10'
        },
    },
}

asr_static_pretrained_models = {
    "deepspeech2offline_aishell-zh-16k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_offline_aishell_ckpt_1.0.1.model.tar.gz',
            'md5':
            '4d26066c6f19f52087425dc722ae5b13',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2/checkpoints/avg_10',
            'model':
            'exp/deepspeech2/checkpoints/avg_10.jit.pdmodel',
            'params':
            'exp/deepspeech2/checkpoints/avg_10.jit.pdiparams',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
            'lm_md5':
            '29e02312deb2e59b3c8686c7966d4fe3'
        }
    },
    "deepspeech2online_aishell-zh-16k": {
        '1.0.1': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_fbank161_ckpt_1.0.1.model.tar.gz',
            'md5':
            'df5ddeac8b679a470176649ac4b78726',
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
        '1.0.2': {
            'url':
            'http://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_fbank161_ckpt_1.0.2.model.tar.gz',
            'md5':
            '4dd42cfce9aaa54db0ec698da6c48ec5',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2_online/checkpoints/avg_1',
            'model':
            'exp/deepspeech2_online/checkpoints/avg_1.jit.pdmodel',
            'params':
            'exp/deepspeech2_online/checkpoints/avg_1.jit.pdiparams',
            'onnx_model':
            'onnx/model.onnx',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
            'lm_md5':
            '29e02312deb2e59b3c8686c7966d4fe3'
        },
    },
    "deepspeech2online_wenetspeech-zh-16k": {
        '1.0.3': {
            'url':
            'http://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr0/asr0_deepspeech2_online_wenetspeech_ckpt_1.0.3.model.tar.gz',
            'md5':
            'cfe273793e68f790f742b411c98bc75e',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2_online/checkpoints/avg_10',
            'model':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdmodel',
            'params':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdiparams',
            'onnx_model':
            'onnx/model.onnx',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
            'lm_md5':
            '29e02312deb2e59b3c8686c7966d4fe3'
        },
        '1.0.4': {
            'url':
            'http://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr0/asr0_deepspeech2_online_wenetspeech_ckpt_1.0.4.model.tar.gz',
            'md5':
            'c595cb76902b5a5d01409171375989f4',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2_online/checkpoints/avg_10',
            'model':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdmodel',
            'params':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdiparams',
            'onnx_model':
            'onnx/model.onnx',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
            'lm_md5':
            '29e02312deb2e59b3c8686c7966d4fe3'
        },
    },
}

asr_onnx_pretrained_models = {
    "deepspeech2online_aishell-zh-16k": {
        '1.0.2': {
            'url':
            'http://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_fbank161_ckpt_1.0.2.model.tar.gz',
            'md5':
            '4dd42cfce9aaa54db0ec698da6c48ec5',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2_online/checkpoints/avg_1',
            'model':
            'exp/deepspeech2_online/checkpoints/avg_1.jit.pdmodel',
            'params':
            'exp/deepspeech2_online/checkpoints/avg_1.jit.pdiparams',
            'onnx_model':
            'onnx/model.onnx',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
            'lm_md5':
            '29e02312deb2e59b3c8686c7966d4fe3'
        },
    },
    "deepspeech2online_wenetspeech-zh-16k": {
        '1.0.3': {
            'url':
            'http://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr0/asr0_deepspeech2_online_wenetspeech_ckpt_1.0.3.model.tar.gz',
            'md5':
            'cfe273793e68f790f742b411c98bc75e',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2_online/checkpoints/avg_10',
            'model':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdmodel',
            'params':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdiparams',
            'onnx_model':
            'onnx/model.onnx',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
            'lm_md5':
            '29e02312deb2e59b3c8686c7966d4fe3'
        },
        '1.0.4': {
            'url':
            'http://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr0/asr0_deepspeech2_online_wenetspeech_ckpt_1.0.4.model.tar.gz',
            'md5':
            'c595cb76902b5a5d01409171375989f4',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2_online/checkpoints/avg_10',
            'model':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdmodel',
            'params':
            'exp/deepspeech2_online/checkpoints/avg_10.jit.pdiparams',
            'onnx_model':
            'onnx/model.onnx',
            'lm_url':
            'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm',
            'lm_md5':
            '29e02312deb2e59b3c8686c7966d4fe3'
        },
    },
}

whisper_dynamic_pretrained_models = {
    "whisper-large-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-large-model.tar.gz',
            'md5':
            'cf1557af9d8ffa493fefad9cb08ae189',
            'cfg_path':
            'whisper.yaml',
            'ckpt_path':
            'whisper-large-model',
            'model':
            'whisper-large-model.pdparams',
            'params':
            'whisper-large-model.pdparams',
            'resource_data':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221108/assets.tar',
            'resource_data_md5':
            '37a0a8abdb3641a51194f79567a93b61',
        },
    },
    "whisper-base-en-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-base-en-model.tar.gz',
            'md5':
            'b156529aefde6beb7726d2ea98fd067a',
            'cfg_path':
            'whisper.yaml',
            'ckpt_path':
            'whisper-base-en-model',
            'model':
            'whisper-base-en-model.pdparams',
            'params':
            'whisper-base-en-model.pdparams',
            'resource_data':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221108/assets.tar',
            'resource_data_md5':
            '37a0a8abdb3641a51194f79567a93b61',
        },
    },
    "whisper-base-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-base-model.tar.gz',
            'md5':
            '6b012a5abd583db14398c3492e47120b',
            'cfg_path':
            'whisper.yaml',
            'ckpt_path':
            'whisper-base-model',
            'model':
            'whisper-base-model.pdparams',
            'params':
            'whisper-base-model.pdparams',
            'resource_data':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221108/assets.tar',
            'resource_data_md5':
            '37a0a8abdb3641a51194f79567a93b61',
        },
    },
    "whisper-medium-en-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-medium-en-model.tar.gz',
            'md5':
            'c7f57d270bd20c7b170ba9dcf6c16f74',
            'cfg_path':
            'whisper.yaml',
            'ckpt_path':
            'whisper-medium-en-model',
            'model':
            'whisper-medium-en-model.pdparams',
            'params':
            'whisper-medium-en-model.pdparams',
            'resource_data':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221108/assets.tar',
            'resource_data_md5':
            '37a0a8abdb3641a51194f79567a93b61',
        },
    },
    "whisper-medium-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-medium-model.tar.gz',
            'md5':
            '4c7dcd0df25f408199db4a4548336786',
            'cfg_path':
            'whisper.yaml',
            'ckpt_path':
            'whisper-medium-model',
            'model':
            'whisper-medium-model.pdparams',
            'params':
            'whisper-medium-model.pdparams',
            'resource_data':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221108/assets.tar',
            'resource_data_md5':
            '37a0a8abdb3641a51194f79567a93b61',
        },
    },
    "whisper-small-en-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-small-en-model.tar.gz',
            'md5':
            '2b24efcb2e93f3275af7c0c7f598ff1c',
            'cfg_path':
            'whisper.yaml',
            'ckpt_path':
            'whisper-small-en-model',
            'model':
            'whisper-small-en-model.pdparams',
            'params':
            'whisper-small-en-model.pdparams',
            'resource_data':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221108/assets.tar',
            'resource_data_md5':
            '37a0a8abdb3641a51194f79567a93b61',
        },
    },
    "whisper-small-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-small-model.tar.gz',
            'md5':
            '5a57911dd41651dd6ed78c5763912825',
            'cfg_path':
            'whisper.yaml',
            'ckpt_path':
            'whisper-small-model',
            'model':
            'whisper-small-model.pdparams',
            'params':
            'whisper-small-model.pdparams',
            'resource_data':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221108/assets.tar',
            'resource_data_md5':
            '37a0a8abdb3641a51194f79567a93b61',
        },
    },
    "whisper-tiny-en-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-tiny-en-model.tar.gz',
            'md5':
            '14969164a3f713fd58e56978c34188f6',
            'cfg_path':
            'whisper.yaml',
            'ckpt_path':
            'whisper-tiny-en-model',
            'model':
            'whisper-tiny-en-model.pdparams',
            'params':
            'whisper-tiny-en-model.pdparams',
            'resource_data':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221108/assets.tar',
            'resource_data_md5':
            '37a0a8abdb3641a51194f79567a93b61',
        },
    },
    "whisper-tiny-16k": {
        '1.3': {
            'url':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221122/whisper-tiny-model.tar.gz',
            'md5':
            'a5b82a1f2067a2ca400f17fabd62b81b',
            'cfg_path':
            'whisper.yaml',
            'ckpt_path':
            'whisper-tiny-model',
            'model':
            'whisper-tiny-model.pdparams',
            'params':
            'whisper-tiny-model.pdparams',
            'resource_data':
            'https://paddlespeech.bj.bcebos.com/whisper/whisper_model_20221108/assets.tar',
            'resource_data_md5':
            '37a0a8abdb3641a51194f79567a93b61',
        },
    },
}

# ---------------------------------
# -------------- CLS --------------
# ---------------------------------
cls_dynamic_pretrained_models = {
    "panns_cnn6-32k": {
        '1.0': {
            'url': 'https://paddlespeech.bj.bcebos.com/cls/panns_cnn6.tar.gz',
            'md5': '4cf09194a95df024fd12f84712cf0f9c',
            'cfg_path': 'panns.yaml',
            'ckpt_path': 'cnn6.pdparams',
            'label_file': 'audioset_labels.txt',
        },
    },
    "panns_cnn10-32k": {
        '1.0': {
            'url': 'https://paddlespeech.bj.bcebos.com/cls/panns_cnn10.tar.gz',
            'md5': 'cb8427b22176cc2116367d14847f5413',
            'cfg_path': 'panns.yaml',
            'ckpt_path': 'cnn10.pdparams',
            'label_file': 'audioset_labels.txt',
        },
    },
    "panns_cnn14-32k": {
        '1.0': {
            'url': 'https://paddlespeech.bj.bcebos.com/cls/panns_cnn14.tar.gz',
            'md5': 'e3b9b5614a1595001161d0ab95edee97',
            'cfg_path': 'panns.yaml',
            'ckpt_path': 'cnn14.pdparams',
            'label_file': 'audioset_labels.txt',
        },
    },
}

cls_static_pretrained_models = {
    "panns_cnn6-32k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/cls/inference_model/panns_cnn6_static.tar.gz',
            'md5':
            'da087c31046d23281d8ec5188c1967da',
            'cfg_path':
            'panns.yaml',
            'model_path':
            'inference.pdmodel',
            'params_path':
            'inference.pdiparams',
            'label_file':
            'audioset_labels.txt',
        },
    },
    "panns_cnn10-32k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/cls/inference_model/panns_cnn10_static.tar.gz',
            'md5':
            '5460cc6eafbfaf0f261cc75b90284ae1',
            'cfg_path':
            'panns.yaml',
            'model_path':
            'inference.pdmodel',
            'params_path':
            'inference.pdiparams',
            'label_file':
            'audioset_labels.txt',
        },
    },
    "panns_cnn14-32k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/cls/inference_model/panns_cnn14_static.tar.gz',
            'md5':
            'ccc80b194821274da79466862b2ab00f',
            'cfg_path':
            'panns.yaml',
            'model_path':
            'inference.pdmodel',
            'params_path':
            'inference.pdiparams',
            'label_file':
            'audioset_labels.txt',
        },
    },
}

# ---------------------------------
# -------------- ST ---------------
# ---------------------------------
st_dynamic_pretrained_models = {
    "fat_st_ted-en-zh": {
        '1.0': {
            "url":
            "https://paddlespeech.bj.bcebos.com/s2t/ted_en_zh/st1/st1_transformer_mtl_noam_ted-en-zh_ckpt_0.1.1.model.tar.gz",
            "md5":
            "d62063f35a16d91210a71081bd2dd557",
            "cfg_path":
            "model.yaml",
            "ckpt_path":
            "exp/transformer_mtl_noam/checkpoints/fat_st_ted-en-zh.pdparams",
        },
    },
}

st_kaldi_bins = {
    "url":
    "https://paddlespeech.bj.bcebos.com/s2t/ted_en_zh/st1/kaldi_bins.tar.gz",
    "md5":
    "c0682303b3f3393dbf6ed4c4e35a53eb",
}

# ---------------------------------
# -------------- TEXT -------------
# ---------------------------------
text_dynamic_pretrained_models = {
    "ernie_linear_p7_wudao-punc-zh": {
        '1.0': {
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
        }
    },
    "ernie_linear_p3_wudao-punc-zh": {
        '1.0': {
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
        }
    },
    "ernie_linear_p3_wudao_fast-punc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/text/ernie_linear_p3_wudao_fast-punc-zh.tar.gz',
            'md5':
            'c93f9594119541a5dbd763381a751d08',
            'cfg_path':
            'ckpt/model_config.json',
            'ckpt_path':
            'ckpt/model_state.pdparams',
            'vocab_file':
            'punc_vocab.txt',
        }
    }
}

# ---------------------------------
# -------------- TTS --------------
# ---------------------------------
tts_dynamic_pretrained_models = {
    # speedyspeech
    "speedyspeech_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/speedyspeech/speedyspeech_csmsc_ckpt_0.2.0.zip',
            'md5':
            '6f6fa967b408454b6662c8c00c0027cb',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_30600.pdz',
            'speech_stats':
            'feats_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
            'tones_dict':
            'tone_id_map.txt',
        },
    },
    # fastspeech2
    "fastspeech2_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip',
            'md5':
            '637d28a5e53aa60275612ba4393d5f22',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_76000.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
        },
    },
    "fastspeech2_canton-canton": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_canton_ckpt_1.4.0.zip',
            'md5':
            '504560c082deba82120927627c900374',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_140000.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
        },
    },
    "fastspeech2_ljspeech-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_ljspeech_ckpt_0.5.zip',
            'md5':
            'ffed800c93deaf16ca9b3af89bfcd747',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_100000.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
        },
    },
    "fastspeech2_aishell3-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_aishell3_ckpt_0.4.zip',
            'md5':
            'f4dd4a5f49a4552b77981f544ab3392e',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_96400.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
        },
    },
    "fastspeech2_vctk-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_vctk_ckpt_0.5.zip',
            'md5':
            '743e5024ca1e17a88c5c271db9779ba4',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_66200.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
        },
    },
    "fastspeech2_cnndecoder_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_cnndecoder_csmsc_ckpt_1.0.0.zip',
            'md5':
            '6eb28e22ace73e0ebe7845f86478f89f',
            'config':
            'cnndecoder.yaml',
            'ckpt':
            'snapshot_iter_153000.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
        },
    },
    "fastspeech2_mix-mix": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_csmscljspeech_add-zhen.zip',
            'md5':
            '77d9d4b5a79ed6203339ead7ef6c74f9',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_94000.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
        },
        '2.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_mix_ckpt_0.2.0.zip',
            'md5':
            '1d938e104e972386c8bfcbcc98a91587',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_99200.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
        },
    },
    "fastspeech2_male-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_male_zh_ckpt_1.4.0.zip',
            'md5':
            '43a9f4bc48a91f5a6f53017474e6c788',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_76000.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
        },
    },
    "fastspeech2_male-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_male_en_ckpt_1.4.0.zip',
            'md5':
            'cc9f44f1f20a8173f63e2d1d41ef1a9c',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_100000.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
        },
    },
    "fastspeech2_male-mix": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_male_mix_ckpt_1.4.0.zip',
            'md5':
            '6d48ad60ef0ab2cee89a5d8cfd93dd86',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_177000.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
        },
    },
    # tacotron2
    "tacotron2_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/tacotron2/tacotron2_csmsc_ckpt_0.2.0.zip',
            'md5':
            '0df4b6f0bcbe0d73c5ed6df8867ab91a',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_30600.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
        },
    },
    "tacotron2_ljspeech-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/tacotron2/tacotron2_ljspeech_ckpt_0.2.0.zip',
            'md5':
            '6a5eddd81ae0e81d16959b97481135f3',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_60300.pdz',
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
        },
    },
    # pwgan
    "pwgan_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip',
            'md5':
            '2e481633325b5bdf0a3823c714d2c117',
            'config':
            'pwg_default.yaml',
            'ckpt':
            'pwg_snapshot_iter_400000.pdz',
            'speech_stats':
            'pwg_stats.npy',
        },
    },
    "pwgan_ljspeech-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_ljspeech_ckpt_0.5.zip',
            'md5':
            '53610ba9708fd3008ccaf8e99dacbaf0',
            'config':
            'pwg_default.yaml',
            'ckpt':
            'pwg_snapshot_iter_400000.pdz',
            'speech_stats':
            'pwg_stats.npy',
        },
    },
    "pwgan_aishell3-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip',
            'md5':
            'd7598fa41ad362d62f85ffc0f07e3d84',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_1000000.pdz',
            'speech_stats':
            'feats_stats.npy',
        },
    },
    "pwgan_vctk-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_vctk_ckpt_0.1.1.zip',
            'md5':
            'b3da1defcde3e578be71eb284cb89f2c',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_1500000.pdz',
            'speech_stats':
            'feats_stats.npy',
        },
    },
    "pwgan_male-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_male_ckpt_1.4.0.zip',
            'md5':
            'a443d6253bf9be377f27ae5972a03c65',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_200000.pdz',
            'speech_stats':
            'feats_stats.npy',
        },
    },
    # mb_melgan
    "mb_melgan_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_ckpt_0.1.1.zip',
            'md5':
            'ee5f0604e20091f0d495b6ec4618b90d',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_1000000.pdz',
            'speech_stats':
            'feats_stats.npy',
        },
    },
    # style_melgan
    "style_melgan_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/style_melgan/style_melgan_csmsc_ckpt_0.1.1.zip',
            'md5':
            '5de2d5348f396de0c966926b8c462755',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_1500000.pdz',
            'speech_stats':
            'feats_stats.npy',
        },
    },
    # hifigan
    "hifigan_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_ckpt_0.1.1.zip',
            'md5':
            'dd40a3d88dfcf64513fba2f0f961ada6',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_2500000.pdz',
            'speech_stats':
            'feats_stats.npy',
        },
    },
    "hifigan_ljspeech-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_ljspeech_ckpt_0.2.0.zip',
            'md5':
            '70e9131695decbca06a65fe51ed38a72',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_2500000.pdz',
            'speech_stats':
            'feats_stats.npy',
        },
    },
    "hifigan_aishell3-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_ckpt_0.2.0.zip',
            'md5':
            '3bb49bc75032ed12f79c00c8cc79a09a',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_2500000.pdz',
            'speech_stats':
            'feats_stats.npy',
        },
    },
    "hifigan_vctk-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_vctk_ckpt_0.2.0.zip',
            'md5':
            '7da8f88359bca2457e705d924cf27bd4',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_2500000.pdz',
            'speech_stats':
            'feats_stats.npy',
        },
    },
    "hifigan_male-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_male_ckpt_1.4.0.zip',
            'md5':
            'a709830596e102c2b83f8adc26d41d85',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_630000.pdz',
            'speech_stats':
            'feats_stats.npy',
        },
    },
    # wavernn
    "wavernn_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/wavernn/wavernn_csmsc_ckpt_0.2.0.zip',
            'md5':
            'ee37b752f09bcba8f2af3b777ca38e13',
            'config':
            'default.yaml',
            'ckpt':
            'snapshot_iter_400000.pdz',
            'speech_stats':
            'feats_stats.npy',
        },
    },
}
tts_dynamic_pretrained_models[
    "fastspeech2_mix-zh"] = tts_dynamic_pretrained_models[
        "fastspeech2_mix-en"] = tts_dynamic_pretrained_models[
            "fastspeech2_mix-mix"]
tts_dynamic_pretrained_models["pwgan_male-en"] = tts_dynamic_pretrained_models[
    "pwgan_male-mix"] = tts_dynamic_pretrained_models["pwgan_male-zh"]
tts_dynamic_pretrained_models[
    "hifigan_male-en"] = tts_dynamic_pretrained_models[
        "hifigan_male-mix"] = tts_dynamic_pretrained_models["hifigan_male-zh"]

tts_static_pretrained_models = {
    # speedyspeech
    "speedyspeech_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/speedyspeech/speedyspeech_nosil_baker_static_0.5.zip',
            'md5':
            'f10cbdedf47dc7a9668d2264494e1823',
            'model':
            'speedyspeech_csmsc.pdmodel',
            'params':
            'speedyspeech_csmsc.pdiparams',
            'phones_dict':
            'phone_id_map.txt',
            'tones_dict':
            'tone_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    # fastspeech2
    "fastspeech2_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_static_0.4.zip',
            'md5':
            '9788cd9745e14c7a5d12d32670b2a5a7',
            'model':
            'fastspeech2_csmsc.pdmodel',
            'params':
            'fastspeech2_csmsc.pdiparams',
            'phones_dict':
            'phone_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_ljspeech-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_ljspeech_static_1.1.0.zip',
            'md5':
            'c49f70b52973423ec45aaa6184fb5bc6',
            'model':
            'fastspeech2_ljspeech.pdmodel',
            'params':
            'fastspeech2_ljspeech.pdiparams',
            'phones_dict':
            'phone_id_map.txt',
            'sample_rate':
            22050,
        },
    },
    "fastspeech2_aishell3-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_aishell3_static_1.1.0.zip',
            'md5':
            '695af44679f48eb4abc159977ddaee16',
            'model':
            'fastspeech2_aishell3.pdmodel',
            'params':
            'fastspeech2_aishell3.pdiparams',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_vctk-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_vctk_static_1.1.0.zip',
            'md5':
            '92d8c082f180bda2fd05a534fb4a1b62',
            'model':
            'fastspeech2_vctk.pdmodel',
            'params':
            'fastspeech2_vctk.pdiparams',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_mix-mix": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_csmscljspeech_add-zhen_static.zip',
            'md5':
            'b5001f66cccafdde07707e1b6269fa58',
            'model':
            'fastspeech2_mix.pdmodel',
            'params':
            'fastspeech2_mix.pdiparams',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
            'sample_rate':
            24000,
        },
        '2.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_mix_static_0.2.0.zip',
            'md5':
            'c6dd138fab3ba261299c0b2efee51d5a',
            'model':
            'fastspeech2_mix.pdmodel',
            'params':
            'fastspeech2_mix.pdiparams',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_male-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_male_zh_static_1.4.0.zip',
            'md5':
            '9b7218829e7fa01aa33dbb2c5f6ef20f',
            'model':
            'fastspeech2_male-zh.pdmodel',
            'params':
            'fastspeech2_male-zh.pdiparams',
            'phones_dict':
            'phone_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_male-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_male_en_static_1.4.0.zip',
            'md5':
            '33cea19b6821b371d242969ffd8b6cbf',
            'model':
            'fastspeech2_male-en.pdmodel',
            'params':
            'fastspeech2_male-en.pdiparams',
            'phones_dict':
            'phone_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_male-mix": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_male_mix_static_1.4.0.zip',
            'md5':
            '66585b04c0ced72f3cb82ee85b814d80',
            'model':
            'fastspeech2_male-mix.pdmodel',
            'params':
            'fastspeech2_male-mix.pdiparams',
            'phones_dict':
            'phone_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    # pwgan
    "pwgan_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_static_0.4.zip',
            'md5':
            'e3504aed9c5a290be12d1347836d2742',
            'model':
            'pwgan_csmsc.pdmodel',
            'params':
            'pwgan_csmsc.pdiparams',
            'sample_rate':
            24000,
        },
    },
    "pwgan_ljspeech-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_ljspeech_static_1.1.0.zip',
            'md5':
            '6f457a069da99c6814ac1fb4677281e4',
            'model':
            'pwgan_ljspeech.pdmodel',
            'params':
            'pwgan_ljspeech.pdiparams',
            'sample_rate':
            22050,
        },
    },
    "pwgan_aishell3-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_aishell3_static_1.1.0.zip',
            'md5':
            '199f64010238275fbdacb326a5cf82d1',
            'model':
            'pwgan_aishell3.pdmodel',
            'params':
            'pwgan_aishell3.pdiparams',
            'sample_rate':
            24000,
        },
    },
    "pwgan_vctk-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_vctk_static_1.1.0.zip',
            'md5':
            'ee0fc571ad5a7fbe4ca20e49df22b819',
            'model':
            'pwgan_vctk.pdmodel',
            'params':
            'pwgan_vctk.pdiparams',
            'sample_rate':
            24000,
        },
    },
    "pwgan_male-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_male_static_1.4.0.zip',
            'md5':
            '52a480ad35694b96603e0a92e9fb3f95',
            'model':
            'pwgan_male.pdmodel',
            'params':
            'pwgan_male.pdiparams',
            'sample_rate':
            24000,
        },
    },
    # mb_melgan
    "mb_melgan_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_static_0.1.1.zip',
            'md5':
            'ac6eee94ba483421d750433f4c3b8d36',
            'model':
            'mb_melgan_csmsc.pdmodel',
            'params':
            'mb_melgan_csmsc.pdiparams',
            'sample_rate':
            24000,
        },
    },
    # hifigan
    "hifigan_csmsc-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_static_0.1.1.zip',
            'md5':
            '7edd8c436b3a5546b3a7cb8cff9d5a0c',
            'model':
            'hifigan_csmsc.pdmodel',
            'params':
            'hifigan_csmsc.pdiparams',
            'sample_rate':
            24000,
        },
    },
    "hifigan_ljspeech-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_ljspeech_static_1.1.0.zip',
            'md5':
            '8c674e79be7c45f6eda74825316438a0',
            'model':
            'hifigan_ljspeech.pdmodel',
            'params':
            'hifigan_ljspeech.pdiparams',
            'sample_rate':
            22050,
        },
    },
    "hifigan_aishell3-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_static_1.1.0.zip',
            'md5':
            '7a10ec5d8d851e2000128f040d30cc01',
            'model':
            'hifigan_aishell3.pdmodel',
            'params':
            'hifigan_aishell3.pdiparams',
            'sample_rate':
            24000,
        },
    },
    "hifigan_vctk-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_vctk_static_1.1.0.zip',
            'md5':
            '130f791dfac84ccdd44ccbdfb67bf08e',
            'model':
            'hifigan_vctk.pdmodel',
            'params':
            'hifigan_vctk.pdiparams',
            'sample_rate':
            24000,
        },
    },
    "hifigan_male-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_male_static_1.4.0.zip',
            'md5':
            '9011fa2738b501e909d1a61054bed29b',
            'model':
            'hifigan_male.pdmodel',
            'params':
            'hifigan_male.pdiparams',
            'sample_rate':
            24000,
        },
    },
}

tts_static_pretrained_models[
    "fastspeech2_mix-zh"] = tts_static_pretrained_models[
        "fastspeech2_mix-en"] = tts_static_pretrained_models[
            "fastspeech2_mix-mix"]
tts_static_pretrained_models["pwgan_male-en"] = tts_static_pretrained_models[
    "pwgan_male-mix"] = tts_static_pretrained_models["pwgan_male-zh"]
tts_static_pretrained_models["hifigan_male-en"] = tts_static_pretrained_models[
    "hifigan_male-mix"] = tts_static_pretrained_models["hifigan_male-zh"]

tts_onnx_pretrained_models = {
    # speedyspeech
    "speedyspeech_csmsc_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/speedyspeech/speedyspeech_csmsc_onnx_0.2.0.zip',
            'md5':
            '3e9c45af9ef70675fc1968ed5074fc88',
            'ckpt':
            'speedyspeech_csmsc.onnx',
            'phones_dict':
            'phone_id_map.txt',
            'tones_dict':
            'tone_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    # fastspeech2
    "fastspeech2_csmsc_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_csmsc_onnx_0.2.0.zip',
            'md5':
            'fd3ad38d83273ad51f0ea4f4abf3ab4e',
            'ckpt':
            'fastspeech2_csmsc.onnx',
            'phones_dict':
            'phone_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_ljspeech_onnx-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_ljspeech_onnx_1.1.0.zip',
            'md5':
            '00754307636a48c972a5f3e65cda3d18',
            'ckpt':
            'fastspeech2_ljspeech.onnx',
            'phones_dict':
            'phone_id_map.txt',
            'sample_rate':
            22050,
        },
    },
    "fastspeech2_aishell3_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_aishell3_onnx_1.1.0.zip',
            'md5':
            'a1d6ee21de897ce394f5469e2bb4df0d',
            'ckpt':
            'fastspeech2_aishell3.onnx',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_vctk_onnx-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_vctk_onnx_1.1.0.zip',
            'md5':
            'd9c3a9b02204a2070504dd99f5f959bf',
            'ckpt':
            'fastspeech2_vctk.onnx',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_cnndecoder_csmsc_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_cnndecoder_csmsc_streaming_onnx_1.0.0.zip',
            'md5':
            '5f70e1a6bcd29d72d54e7931aa86f266',
            'ckpt': [
                'fastspeech2_csmsc_am_encoder_infer.onnx',
                'fastspeech2_csmsc_am_decoder.onnx',
                'fastspeech2_csmsc_am_postnet.onnx',
            ],
            'speech_stats':
            'speech_stats.npy',
            'phones_dict':
            'phone_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_mix_onnx-mix": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_csmscljspeech_add-zhen_onnx.zip',
            'md5':
            '73052520202957920cf54700980933d0',
            'ckpt':
            'fastspeech2_mix.onnx',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
            'sample_rate':
            24000,
        },
        '2.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_mix_onnx_0.2.0.zip',
            'md5':
            '43b8ca5f85709c503777f808eb02a39e',
            'ckpt':
            'fastspeech2_mix.onnx',
            'phones_dict':
            'phone_id_map.txt',
            'speaker_dict':
            'speaker_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_male_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_male_zh_onnx_1.4.0.zip',
            'md5':
            '46c66f5ab86f4fcb493d899d9901c863',
            'ckpt':
            'fastspeech2_male-zh.onnx',
            'phones_dict':
            'phone_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_male_onnx-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_male_en_onnx_1.4.0.zip',
            'md5':
            '401fb5cc31fdb25e22e901c9acba79c8',
            'ckpt':
            'fastspeech2_male-en.onnx',
            'phones_dict':
            'phone_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    "fastspeech2_male_onnx-mix": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_male_mix_onnx_1.4.0.zip',
            'md5':
            '07e51c5991c529b78603034547e9d0fa',
            'ckpt':
            'fastspeech2_male-mix.onnx',
            'phones_dict':
            'phone_id_map.txt',
            'sample_rate':
            24000,
        },
    },
    # pwgan
    "pwgan_csmsc_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_csmsc_onnx_0.2.0.zip',
            'md5':
            '711d0ade33e73f3b721efc9f20669f9c',
            'ckpt':
            'pwgan_csmsc.onnx',
            'sample_rate':
            24000,
        },
    },
    "pwgan_ljspeech_onnx-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_ljspeech_onnx_1.1.0.zip',
            'md5':
            '73cdeeccb77f2ea6ed4d07e71d8ac8b8',
            'ckpt':
            'pwgan_ljspeech.onnx',
            'sample_rate':
            22050,
        },
    },
    "pwgan_aishell3_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_aishell3_onnx_1.1.0.zip',
            'md5':
            '096ab64e152a4fa476aff79ebdadb01b',
            'ckpt':
            'pwgan_aishell3.onnx',
            'sample_rate':
            24000,
        },
    },
    "pwgan_vctk_onnx-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_vctk_onnx_1.1.0.zip',
            'md5':
            '4e754d42cf85f6428f0af887c923d86c',
            'ckpt':
            'pwgan_vctk.onnx',
            'sample_rate':
            24000,
        },
    },
    "pwgan_male_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_male_onnx_1.4.0.zip',
            'md5':
            '13163fd1326f555650dc7141d31767c3',
            'ckpt':
            'pwgan_male.onnx',
            'sample_rate':
            24000,
        },
    },
    # mb_melgan
    "mb_melgan_csmsc_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_onnx_0.2.0.zip',
            'md5':
            '5b83ec746e8414bc29032d954ffd07ec',
            'ckpt':
            'mb_melgan_csmsc.onnx',
            'sample_rate':
            24000,
        },
    },
    # hifigan
    "hifigan_csmsc_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_onnx_0.2.0.zip',
            'md5':
            '1a7dc0385875889e46952e50c0994a6b',
            'ckpt':
            'hifigan_csmsc.onnx',
            'sample_rate':
            24000,
        },
    },
    "hifigan_ljspeech_onnx-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_ljspeech_onnx_1.1.0.zip',
            'md5':
            '062f54b79c1135a50adb5fc8406260b2',
            'ckpt':
            'hifigan_ljspeech.onnx',
            'sample_rate':
            22050,
        },
    },
    "hifigan_aishell3_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_onnx_1.1.0.zip',
            'md5':
            'd6c0d684ad148583ca57837d5e870167',
            'ckpt':
            'hifigan_aishell3.onnx',
            'sample_rate':
            24000,
        },
    },
    "hifigan_vctk_onnx-en": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_vctk_onnx_1.1.0.zip',
            'md5':
            'fd714df3be283c0efbefc8510160ff6d',
            'ckpt':
            'hifigan_vctk.onnx',
            'sample_rate':
            24000,
        },
    },
    "hifigan_male_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_male_onnx_1.4.0.zip',
            'md5':
            'ec6b35417b1fe811d3b1641d4b527769',
            'ckpt':
            'hifigan_male.onnx',
            'sample_rate':
            24000,
        },
    },
}

tts_onnx_pretrained_models[
    "fastspeech2_mix_onnx-zh"] = tts_onnx_pretrained_models[
        "fastspeech2_mix_onnx-en"] = tts_onnx_pretrained_models[
            "fastspeech2_mix_onnx-mix"]
tts_onnx_pretrained_models["pwgan_male_onnx-en"] = tts_onnx_pretrained_models[
    "pwgan_male_onnx-mix"] = tts_onnx_pretrained_models["pwgan_male_onnx-zh"]
tts_onnx_pretrained_models["hifigan_male_onnx-en"] = tts_onnx_pretrained_models[
    "hifigan_male_onnx-mix"] = tts_onnx_pretrained_models[
        "hifigan_male_onnx-zh"]

# ---------------------------------
# ------------ Vector -------------
# ---------------------------------
vector_dynamic_pretrained_models = {
    "ecapatdnn_voxceleb12-16k": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/vector/voxceleb/sv0_ecapa_tdnn_voxceleb12_ckpt_0_2_0.tar.gz',
            'md5':
            'cc33023c54ab346cd318408f43fcaf95',
            'cfg_path':
            'conf/model.yaml',  # the yaml config path
            'ckpt_path':
            'model/model',  # the format is ${dir}/{model_name},
            # so the first 'model' is dir, the second 'model' is the name
            # this means we have a model stored as model/model.pdparams
        },
    },
}

# ---------------------------------
# ------------- KWS ---------------
# ---------------------------------
kws_dynamic_pretrained_models = {
    'mdtc_heysnips-16k': {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/kws/heysnips/kws0_mdtc_heysnips_ckpt.tar.gz',
            'md5':
            'c0de0a9520d66c3c8d6679460893578f',
            'cfg_path':
            'conf/mdtc.yaml',
            'ckpt_path':
            'ckpt/model',
        },
    },
}

# ---------------------------------
# ------------- G2PW ---------------
# ---------------------------------
g2pw_onnx_models = {
    'G2PWModel': {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.0.zip',
            'md5':
            '7e049a55547da840502cf99e8a64f20e',
        },
        '1.1': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip',
            'md5':
            'f8b60501770bff92ed6ce90860a610e6',
        },
    },
}

# ---------------------------------
# ------------- Rhy_frontend ---------------
# ---------------------------------
rhy_frontend_models = {
    'rhy_e2e': {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Rhy_e2e/rhy_frontend.zip',
            'md5': '6624a77393de5925d5a84400b363d8ef',
        },
    },
}
