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
from yacs.config import CfgNode as CN

from paddlespeech.s2t.models.u2 import U2Model
from paddlespeech.s2t.utils.layer_tools import summary


class TestU2Model(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')

        self.batch_size = 2
        self.feat_dim = 83
        self.max_len = 64
        self.vocab_size = 4239

        #(B, T, D)
        audio = np.random.randn(self.batch_size, self.max_len, self.feat_dim)
        audio_len = np.random.randint(self.max_len, size=self.batch_size)
        audio_len[-1] = self.max_len
        #(B, U)
        text = np.array([[1, 2], [1, 2]])
        text_len = np.array([2] * self.batch_size)

        self.audio = paddle.to_tensor(audio, dtype='float32')
        self.audio_len = paddle.to_tensor(audio_len, dtype='int64')
        self.text = paddle.to_tensor(text, dtype='int32')
        self.text_len = paddle.to_tensor(text_len, dtype='int64')

    def test_transformer(self):
        conf_str = """
            # network architecture
            # encoder related
            encoder: transformer
            encoder_conf:
                output_size: 256    # dimension of attention
                attention_heads: 4
                linear_units: 2048  # the number of units of position-wise feed forward
                num_blocks: 12      # the number of encoder blocks
                dropout_rate: 0.1
                positional_dropout_rate: 0.1
                attention_dropout_rate: 0.0
                input_layer: conv2d # encoder architecture type
                normalize_before: true

            # decoder related
            decoder: transformer
            decoder_conf:
                attention_heads: 4
                linear_units: 2048
                num_blocks: 6
                dropout_rate: 0.1
                positional_dropout_rate: 0.1
                self_attention_dropout_rate: 0.0
                src_attention_dropout_rate: 0.0

            # hybrid CTC/attention
            model_conf:
                ctc_weight: 0.3
                lsm_weight: 0.1     # label smoothing option
                length_normalized_loss: false
        """
        cfg = CN().load_cfg(conf_str)
        cfg.input_dim = self.feat_dim
        cfg.output_dim = self.vocab_size
        cfg.cmvn_file = None
        cfg.cmvn_file_type = 'npz'
        cfg.freeze()
        model = U2Model(cfg)
        summary(model, None)
        total_loss, attention_loss, ctc_loss = model(self.audio, self.audio_len,
                                                     self.text, self.text_len)
        self.assertEqual(total_loss.numel(), 1)
        self.assertEqual(attention_loss.numel(), 1)
        self.assertEqual(ctc_loss.numel(), 1)

    def test_conformer(self):
        conf_str = """
            # network architecture
            # encoder related
            encoder: conformer
            encoder_conf:
                output_size: 256    # dimension of attention
                attention_heads: 4
                linear_units: 2048  # the number of units of position-wise feed forward
                num_blocks: 12      # the number of encoder blocks
                dropout_rate: 0.1
                positional_dropout_rate: 0.1
                attention_dropout_rate: 0.0
                input_layer: conv2d # encoder input type, you can chose conv2d, conv2d6 and conv2d8
                normalize_before: true
                cnn_module_kernel: 15
                use_cnn_module: True
                activation_type: 'swish'
                pos_enc_layer_type: 'rel_pos'
                selfattention_layer_type: 'rel_selfattn'

            # decoder related
            decoder: transformer
            decoder_conf:
                attention_heads: 4
                linear_units: 2048
                num_blocks: 6
                dropout_rate: 0.1
                positional_dropout_rate: 0.1
                self_attention_dropout_rate: 0.0
                src_attention_dropout_rate: 0.0

            # hybrid CTC/attention
            model_conf:
                ctc_weight: 0.3
                lsm_weight: 0.1     # label smoothing option
                length_normalized_loss: false
        """
        cfg = CN().load_cfg(conf_str)
        cfg.input_dim = self.feat_dim
        cfg.output_dim = self.vocab_size
        cfg.cmvn_file = None
        cfg.cmvn_file_type = 'npz'
        cfg.freeze()
        model = U2Model(cfg)
        summary(model, None)
        total_loss, attention_loss, ctc_loss = model(self.audio, self.audio_len,
                                                     self.text, self.text_len)
        self.assertEqual(total_loss.numel(), 1)
        self.assertEqual(attention_loss.numel(), 1)
        self.assertEqual(ctc_loss.numel(), 1)


if __name__ == '__main__':
    unittest.main()
