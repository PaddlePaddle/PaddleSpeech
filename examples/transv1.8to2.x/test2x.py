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
"""Evaluation for DeepSpeech2 model."""
import argparse
import functools

import numpy as np
import paddle
import paddle.fluid as fluid
from data_utils.data import DataGenerator
from model_utils.model_check import check_cuda
from model_utils.model_check import check_version

from deepspeech.models.ds2 import DeepSpeech2Model as DS2
from utils.error_rate import char_errors
from utils.error_rate import word_errors
from utils.utility import add_arguments
from utils.utility import print_arguments
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,    128,    "Minibatch size.")
add_arg('beam_size',        int,    500,    "Beam search width.")
add_arg('feat_dim',        int,    161,    "Feature dim.")
add_arg('num_proc_bsearch', int,    8,      "# of CPUs for beam search.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('alpha',            float,  2.5,    "Coef of LM for beam search.")
add_arg('beta',             float,  0.3,    "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  1.0,    "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('share_rnn_weights',            bool,   True,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('test_manifest',   str,
        'data/librispeech/manifest.test-clean',
        "Filepath of manifest to evaluate.")
add_arg('mean_std_path',    str,
        'data/librispeech/mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'data/librispeech/vocab.txt',
        "Filepath of vocabulary.")
add_arg('model_path',       str,
        './checkpoints/libri/step_final',
        "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('lang_model_path',  str,
        'models/lm/common_crawl_00.prune01111.trie.klm',
        "Filepath for language model.")
add_arg('decoding_method',  str,
        'ctc_beam_search',
        "Decoding method. Options: ctc_beam_search, ctc_greedy",
        choices=['ctc_beam_search', 'ctc_greedy'])
add_arg('error_rate_type',  str,
        'wer',
        "Error rate type for evaluation.",
        choices=['wer', 'cer'])
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
# yapf: disable
args = parser.parse_args()

def evaluate():
    """Evaluate on whole test data for DeepSpeech2."""

    # check if set use_gpu=True in paddlepaddle cpu version
    check_cuda(args.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    data_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        keep_transcription_text=True,
        place=place,
        is_training=False)
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.test_manifest,
        batch_size=args.batch_size,
        sortagrad=False,
        shuffle_method=None)


    # decoders only accept string encoded in utf-8
    vocab_list = [chars for chars in data_generator.vocab_list]
    for i, char in enumerate(vocab_list):
        if vocab_list[i] == '':
            vocab_list[i] = " "

    model = DS2(
        feat_size=args.feat_dim,
        dict_size=len(vocab_list),
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        share_rnn_weights=args.share_rnn_weights,
        blank_id=len(vocab_list) - 1
    )

    params_path = args.model_path
    model_dict = paddle.load(params_path)
    model.set_state_dict(model_dict)
    model.eval()
    errors_func = char_errors if args.error_rate_type == 'cer' else word_errors
    errors_sum, len_refs, num_ins = 0.0, 0, 0

    print("start evaluation ...")
    for infer_data in batch_reader():
        audio, target_transcripts, audio_len, mask = infer_data
        audio = np.transpose(audio, (0, 2, 1))
        audio_len = audio_len.reshape(-1)
        audio = paddle.to_tensor(audio)
        audio_len = paddle.to_tensor(audio_len)
        result_transcripts = model.decode(
            audio=audio,
            audio_len=audio_len,
            lang_model_path=args.lang_model_path,
            decoding_method=args.decoding_method,
            beam_alpha=args.alpha,
            beam_beta=args.beta,
            beam_size=args.beam_size,
            cutoff_prob=args.cutoff_prob,
            cutoff_top_n=args.cutoff_top_n,
            vocab_list=vocab_list,
            num_processes=args.num_proc_bsearch
        )
        for target, result in zip(target_transcripts, result_transcripts):
            errors, len_ref = errors_func(target, result)
            errors_sum += errors
            len_refs += len_ref
            num_ins += 1
        print("Error rate [%s] (%d/?) = %f" %
              (args.error_rate_type, num_ins, errors_sum / len_refs))
    print("Final error rate [%s] (%d/%d) = %f" %
          (args.error_rate_type, num_ins, num_ins, errors_sum / len_refs))

    print("finish evaluation")

def main():
    print_arguments(args)
    evaluate()


if __name__ == '__main__':
    main()
