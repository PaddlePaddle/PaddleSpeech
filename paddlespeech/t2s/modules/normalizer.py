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
from paddle import nn


class ZScore(nn.Layer):
    # feature last
    def __init__(self, mu, sigma):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

    def forward(self, x):
        # NOTE: to be compatible with paddle's to_static, we must explicitly
        # call multiply, or add, etc, instead of +-*/, etc.
        return paddle.divide(paddle.subtract(x, self.mu), self.sigma)

    def inverse(self, x):
        # NOTE: to be compatible with paddle's to_static, we must explicitly
        # call multiply, or add, etc, instead of +-*/, etc.
        return paddle.add(paddle.multiply(x, self.sigma), self.mu)
