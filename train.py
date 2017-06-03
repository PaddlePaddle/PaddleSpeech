"""
   Trainer for a simplifed version of Baidu DeepSpeech2 model.
"""

import paddle.v2 as paddle
import distutils.util
import argparse
import gzip
import time
import sys
from model import deep_speech2
from audio_data_utils import DataGenerator
import numpy as np

#TODO: add WER metric

parser = argparse.ArgumentParser(
    description='Simplified version of DeepSpeech2 trainer.')
parser.add_argument(
    "--batch_size", default=32, type=int, help="Minibatch size.")
parser.add_argument(
    "--num_passes",
    default=20,
    type=int,
    help="Training pass number. (default: %(default)s)")
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
    "--adam_learning_rate",
    default=5e-4,
    type=float,
    help="Learning rate for ADAM Optimizer. (default: %(default)s)")
parser.add_argument(
    "--use_gpu",
    default=True,
    type=distutils.util.strtobool,
    help="Use gpu or not. (default: %(default)s)")
parser.add_argument(
    "--use_sortagrad",
    default=False,
    type=distutils.util.strtobool,
    help="Use sortagrad or not. (default: %(default)s)")
parser.add_argument(
    "--trainer_count",
    default=4,
    type=int,
    help="Trainer number. (default: %(default)s)")
parser.add_argument(
    "--normalizer_manifest_path",
    default='data/manifest.libri.train-clean-100',
    type=str,
    help="Manifest path for normalizer. (default: %(default)s)")
parser.add_argument(
    "--train_manifest_path",
    default='data/manifest.libri.train-clean-100',
    type=str,
    help="Manifest path for training. (default: %(default)s)")
parser.add_argument(
    "--dev_manifest_path",
    default='data/manifest.libri.dev-clean',
    type=str,
    help="Manifest path for validation. (default: %(default)s)")
parser.add_argument(
    "--vocab_filepath",
    default='data/eng_vocab.txt',
    type=str,
    help="Vocabulary filepath. (default: %(default)s)")
args = parser.parse_args()


def train():
    """
    DeepSpeech2 training.
    """
    # initialize data generator
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_filepath,
        normalizer_manifest_path=args.normalizer_manifest_path,
        normalizer_num_samples=200,
        max_duration=20.0,
        min_duration=0.0,
        stride_ms=10,
        window_ms=20)

    # create network config
    dict_size = data_generator.vocabulary_size()
    audio_data = paddle.layer.data(
        name="audio_spectrogram",
        height=161,
        width=2000,
        type=paddle.data_type.dense_vector(322000))
    text_data = paddle.layer.data(
        name="transcript_text",
        type=paddle.data_type.integer_value_sequence(dict_size))
    cost = deep_speech2(
        audio_data=audio_data,
        text_data=text_data,
        dict_size=dict_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_size=args.rnn_layer_size,
        is_inference=False)

    # create parameters and optimizer
    parameters = paddle.parameters.create(cost)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.adam_learning_rate, gradient_clipping_threshold=400)
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # prepare data reader
    train_batch_reader_sortagrad = data_generator.batch_reader_creator(
        manifest_path=args.train_manifest_path,
        batch_size=args.batch_size,
        padding_to=2000,
        flatten=True,
        sort_by_duration=True,
        shuffle=False)
    train_batch_reader_nosortagrad = data_generator.batch_reader_creator(
        manifest_path=args.train_manifest_path,
        batch_size=args.batch_size,
        padding_to=2000,
        flatten=True,
        sort_by_duration=False,
        shuffle=True)
    test_batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.dev_manifest_path,
        batch_size=args.batch_size,
        padding_to=2000,
        flatten=True,
        sort_by_duration=False,
        shuffle=False)
    feeding = data_generator.data_name_feeding()

    # create event handler
    def event_handler(event):
        global start_time, cost_sum, cost_counter
        if isinstance(event, paddle.event.EndIteration):
            cost_sum += event.cost
            cost_counter += 1
            if event.batch_id % 50 == 0:
                print "\nPass: %d, Batch: %d, TrainCost: %f" % (
                    event.pass_id, event.batch_id, cost_sum / cost_counter)
                cost_sum, cost_counter = 0.0, 0
                with gzip.open("params.tar.gz", 'w') as f:
                    parameters.to_tar(f)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.BeginPass):
            start_time = time.time()
            cost_sum, cost_counter = 0.0, 0
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_batch_reader, feeding=feeding)
            print "\n------- Time: %d sec,  Pass: %d, ValidationCost: %s" % (
                time.time() - start_time, event.pass_id, result.cost)

    # run train
    # first pass with sortagrad
    if args.use_sortagrad:
        trainer.train(
            reader=train_batch_reader_sortagrad,
            event_handler=event_handler,
            num_passes=1,
            feeding=feeding)
        args.num_passes -= 1
    # other passes without sortagrad
    trainer.train(
        reader=train_batch_reader_nosortagrad,
        event_handler=event_handler,
        num_passes=args.num_passes,
        feeding=feeding)


def main():
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    train()


if __name__ == '__main__':
    main()
