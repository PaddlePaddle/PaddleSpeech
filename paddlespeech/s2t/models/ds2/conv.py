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

from paddlespeech.s2t.modules.subsampling import Conv2dSubsampling4


class Conv2dSubsampling4Pure(Conv2dSubsampling4):
    def __init__(self, idim: int, odim: int, dropout_rate: float):
        super().__init__(idim, odim, dropout_rate, None)
        self.output_dim = ((idim - 1) // 2 - 1) // 2 * odim
        self.receptive_field_length = 2 * (
            3 - 1) + 3  # stride_1 * (kernel_size_2 - 1) + kerel_size_1

    def forward(self, x: paddle.Tensor,
                x_len: paddle.Tensor) -> [paddle.Tensor, paddle.Tensor]:
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        #b, c, t, f = paddle.shape(x) #not work under jit
        x = x.transpose([0, 2, 1, 3]).reshape([0, 0, -1])
        x_len = ((x_len - 1) // 2 - 1) // 2
        return x, x_len
