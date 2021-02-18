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

from yacs.config import CfgNode as CN

_C = CN()
_C.data = CN(
    dict(
        train_manifest="",
        dev_manifest="",
        test_manifest="",
        vocab_filepath="",
        mean_std_filepath="",
        augmentation_config="",
        max_duration=float('inf'),
        min_duration=0.0,
        stride_ms=10.0,  # ms
        window_ms=20.0,  # ms
        n_fft=None,  # fft points
        max_freq=None,  # None for samplerate/2
        specgram_type='linear',  # 'linear', 'mfcc'
        target_sample_rate=16000,  # sample rate
        use_dB_normalization=True,
        target_dB=-20,
        random_seed=0,
        keep_transcription_text=False,
        batch_size=32,  # batch size
        num_workers=0,  # data loader workers
        sortagrad=False,  # sorted in first epoch when True
        shuffle_method="batch_shuffle",  # 'batch_shuffle', 'instance_shuffle'
    ))

_C.model = CN(
    dict(
        num_conv_layers=2,  #Number of stacking convolution layers.
        num_rnn_layers=3,  #Number of stacking RNN layers.
        rnn_layer_size=1024,  #RNN layer size (number of RNN cells).
        use_gru=False,  #Use gru if set True. Use simple rnn if set False.
        share_rnn_weights=True  #Whether to share input-hidden weights between forward and backward directional RNNs.Notice that for GRU, weight sharing is not supported.
    ))

_C.training = CN(
    dict(
        lr=5e-4,  # learning rate
        weight_decay=1e-6,  # the coeff of weight decay
        global_grad_clip=400.0,  # the global norm clip
        plot_interval=1000,  # plot attention and spectrogram by step
        valid_interval=1000,  # validation by step
        save_interval=1000,  # checkpoint by step
        max_iteration=500000,  # max iteration to train by step
        n_epoch=50,  # train epochs
    ))

_C.decoding = CN(
    dict(
        alpha=2.5,  # Coef of LM for beam search.
        beta=0.3,  # Coef of WC for beam search.
        cutoff_prob=1.0,  # Cutoff probability for pruning.
        cutoff_top_n=40,  # Cutoff number for pruning.
        lang_model_path='models/lm/common_crawl_00.prune01111.trie.klm',  # Filepath for language model.
        decoding_method='ctc_beam_search',  # Decoding method. Options: ctc_beam_search, ctc_greedy
        error_rate_type='wer',  # Error rate type for evaluation. Options `wer`, 'cer'
        num_proc_bsearch=8,  # # of CPUs for beam search.
        beam_size=500,  # Beam search width.
        batch_size=128,  # decoding batch size
    ))


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
