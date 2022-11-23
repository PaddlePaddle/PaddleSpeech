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
import unittest

import numpy as np
import paddle

from paddlespeech.s2t.modules.mask import make_non_pad_mask
from paddlespeech.s2t.modules.mask import make_pad_mask


class TestU2Model(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')
        self.lengths = paddle.to_tensor([5, 3, 2])
        self.masks = np.array([
            [True, True, True, True, True],
            [True, True, True, False, False],
            [True, True, False, False, False],
        ])
        self.pad_masks = np.array([
            [False, False, False, False, False],
            [False, False, False, True, True],
            [False, False, True, True, True],
        ])

    def test_make_non_pad_mask(self):
        res = make_non_pad_mask(self.lengths)
        res2 = make_pad_mask(self.lengths).logical_not()
        self.assertSequenceEqual(res.numpy().tolist(), self.masks.tolist())
        self.assertSequenceEqual(res.numpy().tolist(), res2.numpy().tolist())

    def test_make_pad_mask(self):
        res = make_pad_mask(self.lengths)
        res1 = make_non_pad_mask(self.lengths).logical_not()
        self.assertSequenceEqual(res.numpy().tolist(), self.pad_masks.tolist())
        self.assertSequenceEqual(res.numpy().tolist(), res1.tolist())


if __name__ == '__main__':
    unittest.main()
