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
    'cls_dynamic_pretrained_models',
    'cls_static_pretrained_models',
    'st_dynamic_pretrained_models',
    'st_kaldi_bins',
    'text_dynamic_pretrained_models',
    'tts_dynamic_pretrained_models',
    'tts_static_pretrained_models',
    'tts_onnx_pretrained_models',
    'vector_dynamic_pretrained_models',
]

# The tags for pretrained_models should be "{model_name}[_{dataset}][-{lang}][-...]".
# e.g. "conformer_wenetspeech-zh-16k" and "panns_cnn6-32k".
# Command line and python api use "{model_name}[_{dataset}]" as --model, usage:
# "paddlespeech asr --model conformer_wenetspeech --lang zh --sr 16000 --input ./input.wav"

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
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr0/asr0_deepspeech2_online_wenetspeech_ckpt_1.0.2.model.tar.gz',
            'md5':
            'b0c77e7f8881e0a27b82127d1abb8d5f',
            'cfg_path':
            'model.yaml',
            'ckpt_path':
            'exp/deepspeech2_online/checkpoints/avg_10',
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
        '1.0': {
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
        },
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
        },
    },
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
}

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
}

tts_onnx_pretrained_models = {
    # fastspeech2
    "fastspeech2_csmsc_onnx-zh": {
        '1.0': {
            'url':
            'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_csmsc_onnx_0.2.0.zip',
            'md5':
            'fd3ad38d83273ad51f0ea4f4abf3ab4e',
            'ckpt': ['fastspeech2_csmsc.onnx'],
            'phones_dict':
            'phone_id_map.txt',
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
}

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
