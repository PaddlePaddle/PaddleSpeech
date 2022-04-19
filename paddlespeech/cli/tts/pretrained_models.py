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
    # speedyspeech
    "speedyspeech_csmsc-zh": {
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

    # fastspeech2
    "fastspeech2_csmsc-zh": {
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
    "fastspeech2_ljspeech-en": {
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
    "fastspeech2_aishell3-zh": {
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
    "fastspeech2_vctk-en": {
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
    # tacotron2
    "tacotron2_csmsc-zh": {
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
    "tacotron2_ljspeech-en": {
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

    # pwgan
    "pwgan_csmsc-zh": {
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
    "pwgan_ljspeech-en": {
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
    "pwgan_aishell3-zh": {
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
    "pwgan_vctk-en": {
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
    # mb_melgan
    "mb_melgan_csmsc-zh": {
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
    # style_melgan
    "style_melgan_csmsc-zh": {
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
    # hifigan
    "hifigan_csmsc-zh": {
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
    "hifigan_ljspeech-en": {
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
    "hifigan_aishell3-zh": {
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
    "hifigan_vctk-en": {
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

    # wavernn
    "wavernn_csmsc-zh": {
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
    }
}

model_alias = {
    # acoustic model
    "speedyspeech":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeech",
    "speedyspeech_inference":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeechInference",
    "fastspeech2":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2",
    "fastspeech2_inference":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2Inference",
    "tacotron2":
    "paddlespeech.t2s.models.tacotron2:Tacotron2",
    "tacotron2_inference":
    "paddlespeech.t2s.models.tacotron2:Tacotron2Inference",
    # voc
    "pwgan":
    "paddlespeech.t2s.models.parallel_wavegan:PWGGenerator",
    "pwgan_inference":
    "paddlespeech.t2s.models.parallel_wavegan:PWGInference",
    "mb_melgan":
    "paddlespeech.t2s.models.melgan:MelGANGenerator",
    "mb_melgan_inference":
    "paddlespeech.t2s.models.melgan:MelGANInference",
    "style_melgan":
    "paddlespeech.t2s.models.melgan:StyleMelGANGenerator",
    "style_melgan_inference":
    "paddlespeech.t2s.models.melgan:StyleMelGANInference",
    "hifigan":
    "paddlespeech.t2s.models.hifigan:HiFiGANGenerator",
    "hifigan_inference":
    "paddlespeech.t2s.models.hifigan:HiFiGANInference",
    "wavernn":
    "paddlespeech.t2s.models.wavernn:WaveRNN",
    "wavernn_inference":
    "paddlespeech.t2s.models.wavernn:WaveRNNInference",
}
