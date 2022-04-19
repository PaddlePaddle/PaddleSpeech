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
    "panns_cnn6-32k": {
        'url': 'https://paddlespeech.bj.bcebos.com/cls/panns_cnn6.tar.gz',
        'md5': '4cf09194a95df024fd12f84712cf0f9c',
        'cfg_path': 'panns.yaml',
        'ckpt_path': 'cnn6.pdparams',
        'label_file': 'audioset_labels.txt',
    },
    "panns_cnn10-32k": {
        'url': 'https://paddlespeech.bj.bcebos.com/cls/panns_cnn10.tar.gz',
        'md5': 'cb8427b22176cc2116367d14847f5413',
        'cfg_path': 'panns.yaml',
        'ckpt_path': 'cnn10.pdparams',
        'label_file': 'audioset_labels.txt',
    },
    "panns_cnn14-32k": {
        'url': 'https://paddlespeech.bj.bcebos.com/cls/panns_cnn14.tar.gz',
        'md5': 'e3b9b5614a1595001161d0ab95edee97',
        'cfg_path': 'panns.yaml',
        'ckpt_path': 'cnn14.pdparams',
        'label_file': 'audioset_labels.txt',
    },
}

model_alias = {
    "panns_cnn6": "paddlespeech.cls.models.panns:CNN6",
    "panns_cnn10": "paddlespeech.cls.models.panns:CNN10",
    "panns_cnn14": "paddlespeech.cls.models.panns:CNN14",
}
