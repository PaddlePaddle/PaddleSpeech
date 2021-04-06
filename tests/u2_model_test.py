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
import numpy as np
import unittest
from deepspeech.models.u2 import U2TransformerModel
from deepspeech.models.u2 import U2ConformerModel


class TestU2Model(unittest.TestCase):
    def setUp(self):
        batch_size = 2
        feat_dim = 161
        max_len = 100
        audio = np.random.randn(batch_size, feat_dim, max_len)
        audio_len = np.random.randint(100, size=batch_size, dtype='int32')
        audio_len[-1] = 100
        text = np.array([[1, 2], [1, 2]], dtype='int32')
        text_len = np.array([2] * batch_size, dtype='int32')

        self.audio = paddle.to_tensor(audio, dtype='float32')
        self.audio_len = paddle.to_tensor(audio_len, dtype='int64')
        self.text = paddle.to_tensor(text, dtype='int32')
        self.text_len = paddle.to_tensor(text_len, dtype='int64')

        print(audio.shape)
        print(audio_len.shape)
        print(text.shape)
        print(text_len.shape)
        print("-----------------")

    def test_ds2_1(self):
        model = DeepSpeech2Model(
            feat_size=feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            use_gru=False,
            share_rnn_weights=False, )
        logits, probs, logits_len = model(self.audio, self.audio_len, self.text,
                                          self.text_len)
        print('probs.shape', probs.shape)
        print("-----------------")


if __name__ == '__main__':
    unittest.main()
