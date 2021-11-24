# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn


class SoundClassifier(nn.Layer):
    """
    Model for sound classification which uses panns pretrained models to extract
    embeddings from audio files.
    """

    def __init__(self, backbone, num_class, dropout=0.1):
        super(SoundClassifier, self).__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.backbone.emb_size, num_class)

    def forward(self, x):
        # x: (batch_size, num_frames, num_melbins) -> (batch_size, 1, num_frames, num_melbins)
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.dropout(x)
        logits = self.fc(x)

        return logits
