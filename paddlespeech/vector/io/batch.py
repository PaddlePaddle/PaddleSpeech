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
import numpy as np
import paddle


def waveform_collate_fn(batch):
    waveforms = np.stack([item['feat'] for item in batch])
    labels = np.stack([item['label'] for item in batch])

    return {'waveforms': waveforms, 'labels': labels}


def feature_normalize(feats: paddle.Tensor,
                      mean_norm: bool=True,
                      std_norm: bool=True):
    # Features normalization if needed
    mean = feats.mean(axis=-1, keepdim=True) if mean_norm else 0
    std = feats.std(axis=-1, keepdim=True) if std_norm else 1
    feats = (feats - mean) / std

    return feats
