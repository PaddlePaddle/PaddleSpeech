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
import math

import paddle
from paddle import nn
from paddle.jit import to_static
from paddle.static import InputSpec


def test_applicative_evaluation():
    def m_sqrt2(x):
        return paddle.scale(x, math.sqrt(2))

    subgraph = to_static(m_sqrt2, input_spec=[InputSpec([-1])])
    paddle.jit.save(subgraph, './temp_test_to_static')

    fn = paddle.jit.load('./temp_test_to_static')
    x = paddle.arange(10, dtype=paddle.float32)
    y = fn(x)

    print(x)
    print(y)


def test_nested_sequential():
    class Net(nn.Layer):
        def __init__(self):
            super().__init__()
            group1 = nn.Sequential(
                nn.Linear(2, 3),
                nn.Sigmoid(), )
            group2 = nn.Sequential(
                nn.Sequential(nn.Linear(3, 3)),
                nn.Linear(3, 4),
                nn.ReLU(), )
            self.layers = nn.Sequential(group1, group2)

        def forward(self, x):
            return self.layers(x)

    net = Net()
    x = paddle.randn([4, 2])
    y = net(x)
    print(y)

    subgraph = to_static(net, input_spec=[InputSpec([-1, 2])])
    paddle.jit.save(subgraph, './temp_test_to_static')

    fn = paddle.jit.load('./temp_test_to_static')
    y = fn(x)

    print(y)
