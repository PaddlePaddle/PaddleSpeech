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
import random
import argparse
import functools
from time import gmtime, strftime
import socketserver
import struct
import wave
import paddle
import numpy as np

from deepspeech.training.cli import default_argument_parser
from deepspeech.exps.deepspeech2.config import get_cfg_defaults

from deepspeech.frontend.utility import read_manifest
from deepspeech.utils.utility import add_arguments, print_arguments

from deepspeech.models.deepspeech2 import DeepSpeech2Model
from deepspeech.io.dataset import ManifestDataset


class AsrTCPServer(socketserver.TCPServer):
    """The ASR TCP Server."""

    def __init__(self,
                 server_address,
                 RequestHandlerClass,
                 speech_save_dir,
                 audio_process_handler,
                 bind_and_activate=True):
        self.speech_save_dir = speech_save_dir
        self.audio_process_handler = audio_process_handler
        socketserver.TCPServer.__init__(
            self, server_address, RequestHandlerClass, bind_and_activate=True)


class AsrRequestHandler(socketserver.BaseRequestHandler):
    """The ASR request handler."""

    def handle(self):
        # receive data through TCP socket
        chunk = self.request.recv(1024)
        target_len = struct.unpack('>i', chunk[:4])[0]
        data = chunk[4:]
        while len(data) < target_len:
            chunk = self.request.recv(1024)
            data += chunk
        # write to file
        filename = self._write_to_file(data)

        print("Received utterance[length=%d] from %s, saved to %s." %
              (len(data), self.client_address[0], filename))
        start_time = time.time()
        transcript = self.server.audio_process_handler(filename)
        finish_time = time.time()
        print("Response Time: %f, Transcript: %s" %
              (finish_time - start_time, transcript))
        self.request.sendall(transcript.encode('utf-8'))

    def _write_to_file(self, data):
        # prepare save dir and filename
        if not os.path.exists(self.server.speech_save_dir):
            os.mkdir(self.server.speech_save_dir)
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        out_filename = os.path.join(
            self.server.speech_save_dir,
            timestamp + "_" + self.client_address[0] + ".wav")
        # write to wav file
        file = wave.open(out_filename, 'wb')
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(data)
        file.close()
        return out_filename


def warm_up_test(audio_process_handler,
                 manifest_path,
                 num_test_cases,
                 random_seed=0):
    """Warming-up test."""
    manifest = read_manifest(manifest_path)
    rng = random.Random(random_seed)
    samples = rng.sample(manifest, num_test_cases)
    for idx, sample in enumerate(samples):
        print("Warm-up Test Case %d: %s", idx, sample['audio_filepath'])
        start_time = time.time()
        transcript = audio_process_handler(sample['audio_filepath'])
        finish_time = time.time()
        print("Response Time: %f, Transcript: %s" %
              (finish_time - start_time, transcript))


def start_server(config, args):
    """Start the ASR server"""
    dataset = ManifestDataset(
        config.data.test_manifest,
        config.data.vocab_filepath,
        config.data.mean_std_filepath,
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

    model = DeepSpeech2Model(
        feat_size=dataset.feature_size,
        dict_size=dataset.vocab_size,
        num_conv_layers=config.model.num_conv_layers,
        num_rnn_layers=config.model.num_rnn_layers,
        rnn_size=config.model.rnn_layer_size,
        use_gru=config.model.use_gru,
        share_rnn_weights=config.model.share_rnn_weights)
    model.from_pretrained(args.checkpoint_path)
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
