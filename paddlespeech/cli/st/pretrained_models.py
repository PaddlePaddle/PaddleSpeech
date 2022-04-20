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
    "fat_st_ted-en-zh": {
        "url":
        "https://paddlespeech.bj.bcebos.com/s2t/ted_en_zh/st1/st1_transformer_mtl_noam_ted-en-zh_ckpt_0.1.1.model.tar.gz",
        "md5":
        "d62063f35a16d91210a71081bd2dd557",
        "cfg_path":
        "model.yaml",
        "ckpt_path":
        "exp/transformer_mtl_noam/checkpoints/fat_st_ted-en-zh.pdparams",
    }
}

model_alias = {"fat_st": "paddlespeech.s2t.models.u2_st:U2STModel"}

kaldi_bins = {
    "url":
    "https://paddlespeech.bj.bcebos.com/s2t/ted_en_zh/st1/kaldi_bins.tar.gz",
    "md5":
    "c0682303b3f3393dbf6ed4c4e35a53eb",
}
