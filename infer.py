"""Inferer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import distutils.util
import multiprocessing
import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model import deep_speech2
from decoder import ctc_decode
import utils

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--num_samples",
    default=10,
    type=int,
    help="Number of samples for inference. (default: %(default)s)")
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
    "--num_threads_data",
    default=multiprocessing.cpu_count(),
    type=int,
    help="Number of cpu threads for preprocessing data. (default: %(default)s)")
parser.add_argument(
    "--mean_std_filepath",
    default='mean_std.npz',
    type=str,
    help="Manifest path for normalizer. (default: %(default)s)")
parser.add_argument(
    "--decode_manifest_path",
    default='datasets/manifest.test',
    type=str,
    help="Manifest path for decoding. (default: %(default)s)")
parser.add_argument(
    "--model_filepath",
    default='./params.tar.gz',
    type=str,
    help="Model filepath. (default: %(default)s)")
parser.add_argument(
    "--vocab_filepath",
    default='datasets/vocab/eng_vocab.txt',
    type=str,
    help="Vocabulary filepath. (default: %(default)s)")
args = parser.parse_args()


def infer():
    """Max-ctc-decoding for DeepSpeech2."""
    # initialize data generator
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_filepath,
        mean_std_filepath=args.mean_std_filepath,
        augmentation_config='{}',
        num_threads=args.num_threads_data)

    # create network config
    # paddle.data_type.dense_array is used for variable batch input.
    # The size 161 * 161 is only an placeholder value and the real shape
    # of input batch data will be induced during training.
    audio_data = paddle.layer.data(
        name="audio_spectrogram", type=paddle.data_type.dense_array(161 * 161))
    text_data = paddle.layer.data(
        name="transcript_text",
        type=paddle.data_type.integer_value_sequence(data_generator.vocab_size))
    output_probs = deep_speech2(
        audio_data=audio_data,
        text_data=text_data,
        dict_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_size=args.rnn_layer_size,
        is_inference=True)

    # load parameters
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(args.model_filepath))

    # prepare infer data
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.decode_manifest_path,
        batch_size=args.num_samples,
        sortagrad=False,
        shuffle_method=None)
    infer_data = batch_reader().next()

    # run inference
    infer_results = paddle.infer(
        output_layer=output_probs, parameters=parameters, input=infer_data)
    num_steps = len(infer_results) // len(infer_data)
    probs_split = [
        infer_results[i * num_steps:(i + 1) * num_steps]
        for i in xrange(len(infer_data))
    ]

    # decode and print
    for i, probs in enumerate(probs_split):
        output_transcription = ctc_decode(
            probs_seq=probs,
            vocabulary=data_generator.vocab_list,
            method="best_path")
        target_transcription = ''.join(
            [data_generator.vocab_list[index] for index in infer_data[i][1]])
        print("Target Transcription: %s \nOutput Transcription: %s \n" %
              (target_transcription, output_transcription))


def main():
    utils.print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=1)
    infer()


if __name__ == '__main__':
    main()
