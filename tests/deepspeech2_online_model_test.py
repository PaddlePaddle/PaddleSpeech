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

from deepspeech.models.ds2_online import DeepSpeech2ModelOnline


class TestDeepSpeech2ModelOnline(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')

        self.batch_size = 2
        self.feat_dim = 161
        max_len = 210

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
        model = DeepSpeech2ModelOnline(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            num_fc_layers=2,
            fc_layers_size_list=[512, 256],
            use_gru=False)
        loss = model(self.audio, self.audio_len, self.text, self.text_len)
        self.assertEqual(loss.numel(), 1)

    def test_ds2_2(self):
        model = DeepSpeech2ModelOnline(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            num_fc_layers=2,
            fc_layers_size_list=[512, 256],
            use_gru=True)
        loss = model(self.audio, self.audio_len, self.text, self.text_len)
        self.assertEqual(loss.numel(), 1)

    def test_ds2_3(self):
        model = DeepSpeech2ModelOnline(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            num_fc_layers=2,
            fc_layers_size_list=[512, 256],
            use_gru=False)
        loss = model(self.audio, self.audio_len, self.text, self.text_len)
        self.assertEqual(loss.numel(), 1)

    def test_ds2_4(self):
        model = DeepSpeech2ModelOnline(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            num_fc_layers=2,
            fc_layers_size_list=[512, 256],
            use_gru=True)
        loss = model(self.audio, self.audio_len, self.text, self.text_len)
        self.assertEqual(loss.numel(), 1)

    def test_ds2_5(self):
        model = DeepSpeech2ModelOnline(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            num_fc_layers=2,
            fc_layers_size_list=[512, 256],
            use_gru=False)
        loss = model(self.audio, self.audio_len, self.text, self.text_len)
        self.assertEqual(loss.numel(), 1)

    def split_into_chunk(self, x, x_lens, decoder_chunk_size, subsampling_rate,
                         receptive_field_length):
        chunk_size = (decoder_chunk_size - 1
                      ) * subsampling_rate + receptive_field_length
        chunk_stride = subsampling_rate * decoder_chunk_size
        max_len = x.shape[1]
        assert (chunk_size <= max_len)
        x_chunk_list = []
        x_chunk_lens_list = []
        padding_len = chunk_stride - (max_len - chunk_size) % chunk_stride
        padding = paddle.zeros((x.shape[0], padding_len, x.shape[2]))
        padded_x = paddle.concat([x, padding], axis=1)
        num_chunk = (max_len + padding_len - chunk_size) / chunk_stride + 1
        num_chunk = int(num_chunk)
        for i in range(0, num_chunk):
            start = i * chunk_stride
            end = start + chunk_size
            x_chunk = padded_x[:, start:end, :]
            x_len_left = paddle.where(x_lens - i * chunk_stride < 0,
                                      paddle.zeros_like(x_lens),
                                      x_lens - i * chunk_stride)
            x_chunk_len_tmp = paddle.ones_like(x_lens) * chunk_size
            x_chunk_lens = paddle.where(x_len_left < x_chunk_len_tmp,
                                        x_len_left, x_chunk_len_tmp)
            x_chunk_list.append(x_chunk)
            x_chunk_lens_list.append(x_chunk_lens)

        return x_chunk_list, x_chunk_lens_list

    def test_ds2_6(self):
        model = DeepSpeech2ModelOnline(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=1,
            rnn_size=1024,
            num_fc_layers=2,
            fc_layers_size_list=[512, 256],
            use_gru=True)
        model.eval()
        paddle.device.set_device("cpu")
        de_ch_size = 9

        audio_chunk_list, audio_chunk_lens_list = self.split_into_chunk(
            self.audio, self.audio_len, de_ch_size,
            model.encoder.conv.subsampling_rate,
            model.encoder.conv.receptive_field_length)
        eouts_prefix = None
        eouts_lens_prefix = None
        chunk_state_list = [None] * model.encoder.num_rnn_layers
        for i, audio_chunk in enumerate(audio_chunk_list):
            audio_chunk_lens = audio_chunk_lens_list[i]
            probs_pre_chunks, eouts_prefix, eouts_lens_prefix, chunk_state_list = model.decode_prob_by_chunk(
                audio_chunk, audio_chunk_lens, eouts_prefix, eouts_lens_prefix,
                chunk_state_list)
        # print (i, probs_pre_chunks.shape)

        probs, eouts, eouts_lens, final_state_list = model.decode_prob(
            self.audio, self.audio_len)

        decode_max_len = probs.shape[1]
        probs_pre_chunks = probs_pre_chunks[:, :decode_max_len, :]
        self.assertEqual(paddle.allclose(probs, probs_pre_chunks), True)

    def test_ds2_7(self):
        model = DeepSpeech2ModelOnline(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=1,
            rnn_size=1024,
            num_fc_layers=2,
            fc_layers_size_list=[512, 256],
            use_gru=True)
        model.eval()
        paddle.device.set_device("cpu")
        de_ch_size = 9

        probs, eouts, eouts_lens, final_state_list = model.decode_prob(
            self.audio, self.audio_len)
        probs_by_chk, eouts_by_chk, eouts_lens_by_chk, final_state_list_by_chk = model.decode_prob_chunk_by_chunk(
            self.audio, self.audio_len, de_ch_size)
        decode_max_len = probs.shape[1]
        probs_by_chk = probs_by_chk[:, :decode_max_len, :]
        eouts_by_chk = eouts_by_chk[:, :decode_max_len, :]
        self.assertEqual(
            paddle.sum(
                paddle.abs(paddle.subtract(eouts_lens, eouts_lens_by_chk))), 0)
        self.assertEqual(
            paddle.sum(paddle.abs(paddle.subtract(eouts, eouts_by_chk))), 0)
        self.assertEqual(
            paddle.sum(
                paddle.abs(paddle.subtract(probs, probs_by_chk))).numpy(), 0)
        self.assertEqual(paddle.allclose(eouts_by_chk, eouts), True)
        self.assertEqual(paddle.allclose(probs_by_chk, probs), True)
        """
        print ("conv_x", conv_x)
        print ("conv_x_by_chk", conv_x_by_chk)
        print ("final_state_list", final_state_list)
        #print ("final_state_list_by_chk", final_state_list_by_chk)
        print (paddle.sum(paddle.abs(paddle.subtract(eouts[:,:de_ch_size,:], eouts_by_chk[:,:de_ch_size,:]))))
        print (paddle.allclose(eouts[:,:de_ch_size,:], eouts_by_chk[:,:de_ch_size,:]))
        print (paddle.sum(paddle.abs(paddle.subtract(eouts[:,de_ch_size:de_ch_size*2,:], eouts_by_chk[:,de_ch_size:de_ch_size*2,:]))))
        print (paddle.allclose(eouts[:,de_ch_size:de_ch_size*2,:], eouts_by_chk[:,de_ch_size:de_ch_size*2,:]))
        print (paddle.sum(paddle.abs(paddle.subtract(eouts[:,de_ch_size*2:de_ch_size*3,:], eouts_by_chk[:,de_ch_size*2:de_ch_size*3,:]))))
        print (paddle.allclose(eouts[:,de_ch_size*2:de_ch_size*3,:], eouts_by_chk[:,de_ch_size*2:de_ch_size*3,:]))
        print (paddle.sum(paddle.abs(paddle.subtract(eouts, eouts_by_chk))))
        print (paddle.sum(paddle.abs(paddle.subtract(eouts, eouts_by_chk))))
        print (paddle.allclose(eouts[:,:,:], eouts_by_chk[:,:,:]))
        """


if __name__ == '__main__':
    unittest.main()
