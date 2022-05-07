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
# Static model applied on paddle inference
pretrained_models = {
    # speedyspeech
    "speedyspeech_csmsc-zh": {
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
    # fastspeech2
    "fastspeech2_csmsc-zh": {
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
    # pwgan
    "pwgan_csmsc-zh": {
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
    # mb_melgan
    "mb_melgan_csmsc-zh": {
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
    # hifigan
    "hifigan_csmsc-zh": {
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
}
