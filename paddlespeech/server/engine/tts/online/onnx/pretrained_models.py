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

# support online model
pretrained_models = {
    # fastspeech2
    "fastspeech2_csmsc_onnx-zh": {
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
    "fastspeech2_cnndecoder_csmsc_onnx-zh": {
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

    # mb_melgan
    "mb_melgan_csmsc_onnx-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_onnx_0.2.0.zip',
        'md5':
        '5b83ec746e8414bc29032d954ffd07ec',
        'ckpt':
        'mb_melgan_csmsc.onnx',
        'sample_rate':
        24000,
    },

    # hifigan
    "hifigan_csmsc_onnx-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_onnx_0.2.0.zip',
        'md5':
        '1a7dc0385875889e46952e50c0994a6b',
        'ckpt':
        'hifigan_csmsc.onnx',
        'sample_rate':
        24000,
    },
}

