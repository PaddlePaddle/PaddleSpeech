"""Server-end for the ASR demo."""
import os
import time
import random
import argparse
import functools
from time import gmtime, strftime
import SocketServer
import struct
import wave
import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model import DeepSpeech2Model
from data_utils.utils import read_manifest
from utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('host_port',        int,    8086,    "Server's IP port.")
add_arg('beam_size',        int,    500,    "Beam search width.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('alpha',            float,  0.36,   "Coef of LM for beam search.")
add_arg('beta',             float,  0.25,   "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  0.99,   "Cutoff probability for pruning.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of Simple RNNs.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   True,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('host_ip',          str,
        'localhost',
        "Server's IP address.")
add_arg('speech_save_dir',  str,
        'demo_cache',
        "Directory to save demo audios.")
add_arg('warmup_manifest',  str,
        'datasets/manifest.test',
        "Filepath of manifest to warm up.")
add_arg('mean_std_path',    str,
        'mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'datasets/vocab/eng_vocab.txt',
        "Filepath of vocabulary.")
add_arg('model_path',       str,
        './checkpoints/params.latest.tar.gz',
        "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('lang_model_path',  str,
        'lm/data/common_crawl_00.prune01111.trie.klm',
        "Filepath for language model.")
add_arg('decoder_method',   str,
        'ctc_beam_search',
        "Decoder method. Options: ctc_beam_search, ctc_greedy",
        choices = ['ctc_beam_search', 'ctc_greedy'])
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
# yapf: disable
args = parser.parse_args()


class AsrTCPServer(SocketServer.TCPServer):
    """The ASR TCP Server."""

    def __init__(self,
                 server_address,
                 RequestHandlerClass,
                 speech_save_dir,
                 audio_process_handler,
                 bind_and_activate=True):
        self.speech_save_dir = speech_save_dir
        self.audio_process_handler = audio_process_handler
        SocketServer.TCPServer.__init__(
            self, server_address, RequestHandlerClass, bind_and_activate=True)


class AsrRequestHandler(SocketServer.BaseRequestHandler):
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
        self.request.sendall(transcript)

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
        file.setsampwidth(4)
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


def start_server():
    """Start the ASR server"""
    # prepare data generator
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        num_threads=1)
    # prepare ASR model
    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        pretrained_model_path=args.model_path,
        share_rnn_weights=args.share_rnn_weights)

    # prepare ASR inference handler
    def file_to_transcript(filename):
        feature = data_generator.process_utterance(filename, "")
        result_transcript = ds2_model.infer_batch(
            infer_data=[feature],
            decoder_method=args.decoder_method,
            beam_alpha=args.alpha,
            beam_beta=args.beta,
            beam_size=args.beam_size,
            cutoff_prob=args.cutoff_prob,
            vocab_list=data_generator.vocab_list,
            language_model_path=args.lang_model_path,
            num_processes=1)
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


def main():
    print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=1)
    start_server()


if __name__ == "__main__":
    main()
