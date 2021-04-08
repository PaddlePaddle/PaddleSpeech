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
"""Server-end for the ASR demo."""
import os
import time
import argparse
import functools
import paddle
import numpy as np

from deepspeech.utils.socket_server import warm_up_test
from deepspeech.utils.socket_server import AsrTCPServer
from deepspeech.utils.socket_server import AsrRequestHandler

from deepspeech.training.cli import default_argument_parser
from deepspeech.exps.deepspeech2.config import get_cfg_defaults

from deepspeech.frontend.utility import read_manifest
from deepspeech.utils.utility import add_arguments, print_arguments

from deepspeech.models.deepspeech2 import DeepSpeech2Model
from deepspeech.io.dataset import ManifestDataset


def start_server(config, args):
    """Start the ASR server"""
    dataset = ManifestDataset(
        config.data.test_manifest,
        config.data.unit_type,
        config.data.vocab_filepath,
        config.data.mean_std_filepath,
        spm_model_prefix=config.data.spm_model_prefix,
        augmentation_config="{}",
        max_duration=config.data.max_duration,
        min_duration=config.data.min_duration,
        stride_ms=config.data.stride_ms,
        window_ms=config.data.window_ms,
        n_fft=config.data.n_fft,
        max_freq=config.data.max_freq,
        target_sample_rate=config.data.target_sample_rate,
        specgram_type=config.data.specgram_type,
        use_dB_normalization=config.data.use_dB_normalization,
        target_dB=config.data.target_dB,
        random_seed=config.data.random_seed,
        keep_transcription_text=True)
    model = DeepSpeech2Model.from_pretrained(dataset, config,
                                             args.checkpoint_path)
    model.eval()

    # prepare ASR inference handler
    def file_to_transcript(filename):
        feature = dataset.process_utterance(filename, "")
        audio = np.array([feature[0]]).astype('float32')  #[1, D, T]
        audio_len = feature[0].shape[1]
        audio_len = np.array([audio_len]).astype('int64')  # [1]

        result_transcript = model.decode(
            paddle.to_tensor(audio),
            paddle.to_tensor(audio_len),
            vocab_list=dataset.vocab_list,
            decoding_method=config.decoding.decoding_method,
            lang_model_path=config.decoding.lang_model_path,
            beam_alpha=config.decoding.alpha,
            beam_beta=config.decoding.beta,
            beam_size=config.decoding.beam_size,
            cutoff_prob=config.decoding.cutoff_prob,
            cutoff_top_n=config.decoding.cutoff_top_n,
            num_processes=config.decoding.num_proc_bsearch)
        return result_transcript[0]

    # warming up with utterrances sampled from Librispeech
    print('-----------------------------------------------------------')
    print('Warming up ...')
    warm_up_test(
        audio_process_handler=file_to_transcript,
        manifest_path=args.warmup_manifest,
        num_test_cases=3)
    print('-----------------------------------------------------------')

    # start the server
    server = AsrTCPServer(
        server_address=(args.host_ip, args.host_port),
        RequestHandlerClass=AsrRequestHandler,
        speech_save_dir=args.speech_save_dir,
        audio_process_handler=file_to_transcript)
    print("ASR Server Started.")
    server.serve_forever()


def main(config, args):
    start_server(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    add_arg('host_ip',          str,
            'localhost',
            "Server's IP address.")
    add_arg('host_port',        int,    8086,    "Server's IP port.")
    add_arg('speech_save_dir',  str,
            'demo_cache',
            "Directory to save demo audios.")
    add_arg('warmup_manifest',  str, None, "Filepath of manifest to warm up.")
    args = parser.parse_args()
    print_arguments(args)

    # https://yaml.org/type/float.html
    config = get_cfg_defaults()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)

    args.warmup_manifest = config.data.test_manifest
    print_arguments(args)

    if args.dump_config:
        with open(args.dump_config, 'w') as f:
            print(config, file=f)

    main(config, args)
