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
import os
import pickle
import unittest

import numpy as np
import paddle
from paddle import inference

from paddlespeech.s2t.models.ds2 import DeepSpeech2InferModel
from paddlespeech.s2t.models.ds2 import DeepSpeech2Model


class TestDeepSpeech2Model(unittest.TestCase):
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
        model = DeepSpeech2Model(
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
        model = DeepSpeech2Model(
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
        model = DeepSpeech2Model(
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
        model = DeepSpeech2Model(
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
        model = DeepSpeech2Model(
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

    def test_ds2_6(self):
        model = DeepSpeech2Model(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=3,
            rnn_size=1024,
            rnn_direction='bidirect',
            num_fc_layers=2,
            fc_layers_size_list=[512, 256],
            use_gru=False)
        loss = model(self.audio, self.audio_len, self.text, self.text_len)
        self.assertEqual(loss.numel(), 1)

    def test_ds2_7(self):
        use_gru = False
        model = DeepSpeech2Model(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=1,
            rnn_size=1024,
            rnn_direction='forward',
            num_fc_layers=2,
            fc_layers_size_list=[512, 256],
            use_gru=use_gru)
        model.eval()
        paddle.device.set_device("cpu")
        de_ch_size = 8

        eouts, eouts_lens, final_state_h_box, final_state_c_box = model.encoder(
            self.audio, self.audio_len)
        eouts_by_chk_list, eouts_lens_by_chk_list, final_state_h_box_chk, final_state_c_box_chk = model.encoder.forward_chunk_by_chunk(
            self.audio, self.audio_len, de_ch_size)
        eouts_by_chk = paddle.concat(eouts_by_chk_list, axis=1)
        eouts_lens_by_chk = paddle.add_n(eouts_lens_by_chk_list)
        decode_max_len = eouts.shape[1]
        eouts_by_chk = eouts_by_chk[:, :decode_max_len, :]
        self.assertEqual(paddle.allclose(eouts_by_chk, eouts), True)
        self.assertEqual(
            paddle.allclose(final_state_h_box, final_state_h_box_chk), True)
        if use_gru is False:
            self.assertEqual(
                paddle.allclose(final_state_c_box, final_state_c_box_chk), True)

    def test_ds2_8(self):
        use_gru = True
        model = DeepSpeech2Model(
            feat_size=self.feat_dim,
            dict_size=10,
            num_conv_layers=2,
            num_rnn_layers=1,
            rnn_size=1024,
            rnn_direction='forward',
            num_fc_layers=2,
            fc_layers_size_list=[512, 256],
            use_gru=use_gru)
        model.eval()
        paddle.device.set_device("cpu")
        de_ch_size = 8

        eouts, eouts_lens, final_state_h_box, final_state_c_box = model.encoder(
            self.audio, self.audio_len)
        eouts_by_chk_list, eouts_lens_by_chk_list, final_state_h_box_chk, final_state_c_box_chk = model.encoder.forward_chunk_by_chunk(
            self.audio, self.audio_len, de_ch_size)
        eouts_by_chk = paddle.concat(eouts_by_chk_list, axis=1)
        eouts_lens_by_chk = paddle.add_n(eouts_lens_by_chk_list)
        decode_max_len = eouts.shape[1]
        eouts_by_chk = eouts_by_chk[:, :decode_max_len, :]
        self.assertEqual(paddle.allclose(eouts_by_chk, eouts), True)
        self.assertEqual(
            paddle.allclose(final_state_h_box, final_state_h_box_chk), True)
        if use_gru is False:
            self.assertEqual(
                paddle.allclose(final_state_c_box, final_state_c_box_chk), True)


class TestDeepSpeech2StaticModelOnline(unittest.TestCase):
    def setUp(self):
        export_prefix = "exp/deepspeech2_online/checkpoints/test_export"
        if not os.path.exists(os.path.dirname(export_prefix)):
            os.makedirs(os.path.dirname(export_prefix), mode=0o755)
        infer_model = DeepSpeech2InferModel(
            feat_size=161,
            dict_size=4233,
            num_conv_layers=2,
            num_rnn_layers=5,
            rnn_size=1024,
            num_fc_layers=0,
            fc_layers_size_list=[-1],
            use_gru=False)
        static_model = infer_model.export()
        paddle.jit.save(static_model, export_prefix)

        with open("test_data/static_ds2online_inputs.pickle", "rb") as f:
            self.data_dict = pickle.load(f)

        self.setup_model(export_prefix)

    def setup_model(self, export_prefix):
        deepspeech_config = inference.Config(export_prefix + ".pdmodel",
                                             export_prefix + ".pdiparams")
        if ('CUDA_VISIBLE_DEVICES' in os.environ.keys() and
                os.environ['CUDA_VISIBLE_DEVICES'].strip() != ''):
            deepspeech_config.enable_use_gpu(100, 0)
            deepspeech_config.enable_memory_optim()
        deepspeech_predictor = inference.create_predictor(deepspeech_config)
        self.predictor = deepspeech_predictor

    def test_unit(self):
        input_names = self.predictor.get_input_names()
        audio_handle = self.predictor.get_input_handle(input_names[0])
        audio_len_handle = self.predictor.get_input_handle(input_names[1])
        h_box_handle = self.predictor.get_input_handle(input_names[2])
        c_box_handle = self.predictor.get_input_handle(input_names[3])

        x_chunk = self.data_dict["audio_chunk"]
        x_chunk_lens = self.data_dict["audio_chunk_lens"]
        chunk_state_h_box = self.data_dict["chunk_state_h_box"]
        chunk_state_c_box = self.data_dict["chunk_state_c_bos"]

        audio_handle.reshape(x_chunk.shape)
        audio_handle.copy_from_cpu(x_chunk)

        audio_len_handle.reshape(x_chunk_lens.shape)
        audio_len_handle.copy_from_cpu(x_chunk_lens)

        h_box_handle.reshape(chunk_state_h_box.shape)
        h_box_handle.copy_from_cpu(chunk_state_h_box)

        c_box_handle.reshape(chunk_state_c_box.shape)
        c_box_handle.copy_from_cpu(chunk_state_c_box)

        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        output_lens_handle = self.predictor.get_output_handle(output_names[1])
        output_state_h_handle = self.predictor.get_output_handle(
            output_names[2])
        output_state_c_handle = self.predictor.get_output_handle(
            output_names[3])
        self.predictor.run()

        output_chunk_probs = output_handle.copy_to_cpu()
        output_chunk_lens = output_lens_handle.copy_to_cpu()
        chunk_state_h_box = output_state_h_handle.copy_to_cpu()
        chunk_state_c_box = output_state_c_handle.copy_to_cpu()
        return True


if __name__ == '__main__':
    unittest.main()
