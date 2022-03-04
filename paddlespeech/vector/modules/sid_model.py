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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SpeakerIdetification(nn.Layer):
    def __init__(
            self,
            backbone,
            num_class,
            lin_blocks=0,
            lin_neurons=192,
            dropout=0.1, ):

        super(SpeakerIdetification, self).__init__()
        self.backbone = backbone
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        input_size = self.backbone.emb_size
        self.blocks = nn.LayerList()
        for i in range(lin_blocks):
            self.blocks.extend([
                nn.BatchNorm1D(input_size),
                nn.Linear(in_features=input_size, out_features=lin_neurons),
            ])
            input_size = lin_neurons

        self.weight = paddle.create_parameter(
            shape=(input_size, num_class),
            dtype='float32',
            attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()), )

    def forward(self, x, lengths=None):
        # x.shape: (N, C, L)
        x = self.backbone(x, lengths).squeeze(
            -1)  # (N, emb_size, 1) -> (N, emb_size)
        if self.dropout is not None:
            x = self.dropout(x)

        for fc in self.blocks:
            x = fc(x)

        logits = F.linear(F.normalize(x), F.normalize(self.weight, axis=0))

        return logits
