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

from paddlespeech.s2t.models.ds2 import DeepSpeech2Model


class TestDeepSpeech2Model(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')

        self.batch_size = 2
        self.feat_dim = 161
        max_len = 64

        # (B, T, D)
        audio = np.random.randn(self.batch_size, max_len, self.feat_dim)
        audio_len = np.random.randint(max_len, size=self.batch_size)
        audio_len[-1] = max_len
        # (B, U)
        text = np.array([[1, 2], [1, 2]])
        text_len = np.array([2] * self.batch_size)

        self.audio = paddle.to_tensor(audio, dtype='float32')
        self.audio_len = paddle.to_tensor(audio_len, dtype='int64')
        self.text = paddle.to_tensor(text, dtype='int32')
        self.text_len = paddle.to_tensor(text_len, dtype='int64')

    def test_ds2_1(self):
        model = DeepSpeech2Model(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            use_gru=False,
            share_rnn_weights=False, )
        loss = model(self.audio, self.audio_len, self.text, self.text_len)
        self.assertEqual(loss.numel(), 1)

    def test_ds2_2(self):
        model = DeepSpeech2Model(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            use_gru=True,
            share_rnn_weights=False, )
        loss = model(self.audio, self.audio_len, self.text, self.text_len)
        self.assertEqual(loss.numel(), 1)

    def test_ds2_3(self):
        model = DeepSpeech2Model(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            use_gru=False,
            share_rnn_weights=True, )
        loss = model(self.audio, self.audio_len, self.text, self.text_len)
        self.assertEqual(loss.numel(), 1)

    def test_ds2_4(self):
        model = DeepSpeech2Model(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            use_gru=True,
            share_rnn_weights=True, )
        loss = model(self.audio, self.audio_len, self.text, self.text_len)
        self.assertEqual(loss.numel(), 1)

    def test_ds2_5(self):
        model = DeepSpeech2Model(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            use_gru=False,
            share_rnn_weights=False, )
        loss = model(self.audio, self.audio_len, self.text, self.text_len)
        self.assertEqual(loss.numel(), 1)


if __name__ == '__main__':
    unittest.main()
