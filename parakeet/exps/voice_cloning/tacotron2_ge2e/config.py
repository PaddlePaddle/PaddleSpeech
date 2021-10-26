# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from yacs.config import CfgNode as CN

_C = CN()
_C.data = CN(
    dict(
        batch_size=32,  # batch size
        valid_size=64,  # the first N examples are reserved for validation
        sample_rate=22050,  # Hz, sample rate
        n_fft=1024,  # fft frame size
        win_length=1024,  # window size
        hop_length=256,  # hop size between ajacent frame
        fmax=8000,  # Hz, max frequency when converting to mel
        fmin=0,  # Hz, min frequency when converting to mel
        d_mels=80,  # mel bands
        padding_idx=0,  # text embedding's padding index
    ))

_C.model = CN(
    dict(
        vocab_size=70,
        n_tones=10,
        reduction_factor=1,  # reduction factor
        d_encoder=512,  # embedding & encoder's internal size
        encoder_conv_layers=3,  # number of conv layer in tacotron2 encoder
        encoder_kernel_size=5,  # kernel size of conv layers in tacotron2 encoder
        d_prenet=256,  # hidden size of decoder prenet
        # hidden size of the first rnn layer in tacotron2 decoder
        d_attention_rnn=1024,
        # hidden size of the second rnn layer in tacotron2 decoder
        d_decoder_rnn=1024,
        d_attention=128,  # hidden size of  decoder location linear layer
        attention_filters=32,  # number of filter in decoder location conv layer
        attention_kernel_size=31,  # kernel size of decoder location conv layer
        d_postnet=512,  # hidden size of decoder postnet
        postnet_kernel_size=5,  # kernel size of conv layers in postnet
        postnet_conv_layers=5,  # number of conv layer in decoder postnet
        p_encoder_dropout=0.5,  # droput probability in encoder
        p_prenet_dropout=0.5,  # droput probability in decoder prenet

        # droput probability of first rnn layer in decoder
        p_attention_dropout=0.1,
        # droput probability of second rnn layer in decoder
        p_decoder_dropout=0.1,
        p_postnet_dropout=0.5,  # droput probability in decoder postnet
        guided_attention_loss_sigma=0.2,
        d_global_condition=256,

        # whether to use a classifier to predict stop probability
        use_stop_token=False,
        # whether to use guided attention loss in training
        use_guided_attention_loss=True, ))

_C.training = CN(
    dict(
        lr=1e-3,  # learning rate
        weight_decay=1e-6,  # the coeff of weight decay
        grad_clip_thresh=1.0,  # the clip norm of grad clip.
        valid_interval=1000,  # validation
        save_interval=1000,  # checkpoint
        max_iteration=500000,  # max iteration to train
    ))


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
