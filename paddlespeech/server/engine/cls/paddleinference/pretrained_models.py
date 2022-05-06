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
    "panns_cnn6-32k": {
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
    "panns_cnn10-32k": {
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
    "panns_cnn14-32k": {
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
}
