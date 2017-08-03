import os
import time
import random
import argparse
import distutils.util
from time import gmtime, strftime
import SocketServer
import struct
import wave
import pyaudio
import paddle.v2 as paddle
from utils import print_arguments
from data_utils.data import DataGenerator
from model import DeepSpeech2Model
from data_utils.utils import read_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--host_ip",
    default="10.104.18.14",
    type=str,
    help="Server IP address. (default: %(default)s)")
parser.add_argument(
    "--host_port",
    default=8086,
    type=int,
    help="Server Port. (default: %(default)s)")
parser.add_argument(
    "--speech_save_dir",
    default="demo_cache",
    type=str,
    help="Directory for saving demo speech. (default: %(default)s)")
parser.add_argument(
    "--vocab_filepath",
    default='datasets/vocab/eng_vocab.txt',
    type=str,
    help="Vocabulary filepath. (default: %(default)s)")
parser.add_argument(
    "--mean_std_filepath",
    default='mean_std.npz',
    type=str,
    help="Manifest path for normalizer. (default: %(default)s)")
parser.add_argument(
    "--warmup_manifest_path",
    default='datasets/manifest.test',
    type=str,
    help="Manifest path for warmup test. (default: %(default)s)")
parser.add_argument(
    "--specgram_type",
    default='linear',
    type=str,
    help="Feature type of audio data: 'linear' (power spectrum)"
    " or 'mfcc'. (default: %(default)s)")
parser.add_argument(
    "--num_conv_layers",
    default=2,
    type=int,
    help="Convolution layer number. (default: %(default)s)")
parser.add_argument(
    "--num_rnn_layers",
    default=3,
    type=int,
    help="RNN layer number. (default: %(default)s)")
parser.add_argument(
    "--rnn_layer_size",
    default=512,
    type=int,
    help="RNN layer cell number. (default: %(default)s)")
parser.add_argument(
    "--use_gpu",
    default=True,
    type=distutils.util.strtobool,
    help="Use gpu or not. (default: %(default)s)")
parser.add_argument(
    "--model_filepath",
    default='checkpoints/params.latest.tar.gz',
    type=str,
    help="Model filepath. (default: %(default)s)")
parser.add_argument(
    "--decode_method",
    default='beam_search',
    type=str,
    help="Method for ctc decoding: best_path or beam_search. "
    "(default: %(default)s)")
parser.add_argument(
    "--beam_size",
    default=100,
    type=int,
    help="Width for beam search decoding. (default: %(default)d)")
parser.add_argument(
    "--language_model_path",
    default="lm/data/common_crawl_00.prune01111.trie.klm",
    type=str,
    help="Path for language model. (default: %(default)s)")
parser.add_argument(
    "--alpha",
    default=0.36,
    type=float,
    help="Parameter associated with language model. (default: %(default)f)")
parser.add_argument(
    "--beta",
    default=0.25,
    type=float,
    help="Parameter associated with word count. (default: %(default)f)")
parser.add_argument(
    "--cutoff_prob",
    default=0.99,
    type=float,
    help="The cutoff probability of pruning"
    "in beam search. (default: %(default)f)")
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
        vocab_filepath=args.vocab_filepath,
        mean_std_filepath=args.mean_std_filepath,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        num_threads=1)
    # prepare ASR model
    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        pretrained_model_path=args.model_filepath)

    # prepare ASR inference handler
    def file_to_transcript(filename):
        feature = data_generator.process_utterance(filename, "")
        result_transcript = ds2_model.infer_batch(
            infer_data=[feature],
            decode_method=args.decode_method,
            beam_alpha=args.alpha,
            beam_beta=args.beta,
            beam_size=args.beam_size,
            cutoff_prob=args.cutoff_prob,
            vocab_list=data_generator.vocab_list,
            language_model_path=args.language_model_path,
            num_processes=1)
        return result_transcript[0]

    # warming up with utterrances sampled from Librispeech
    print('-----------------------------------------------------------')
    print('Warming up ...')
    warm_up_test(
        audio_process_handler=file_to_transcript,
        manifest_path=args.warmup_manifest_path,
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
