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
    # The tags for pretrained_models should be "{model_name}[-{dataset}][-{sr}][-...]".
    # e.g. "ecapatdnn_voxceleb12-16k".
    # Command line and python api use "{model_name}[-{dataset}]" as --model, usage:
    # "paddlespeech vector --task spk --model ecapatdnn_voxceleb12-16k --sr 16000 --input ./input.wav"
    "ecapatdnn_voxceleb12-16k": {
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
}

model_alias = {
    "ecapatdnn": "paddlespeech.vector.models.ecapa_tdnn:EcapaTdnn",
}
