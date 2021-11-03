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

from paddlespeech.t2s.modules import expansion


def test_expand():
    x = paddle.randn([2, 4, 3])  # (B, T, C)
    lengths = paddle.to_tensor([[1, 2, 2, 1], [3, 1, 4, 0]])
    y = expansion.expand(x, lengths)

    assert y.shape == [2, 8, 3]
    print("the first sequence")
    print(y[0])

    print("the second sequence")
    print(y[1])
