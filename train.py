"""Trainer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import gzip
import time
import distutils.util
import paddle.v2 as paddle
from model import deep_speech2
from data_utils.data import DataGenerator
import utils

parser = argparse.ArgumentParser(description=__doc__)
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
    default=True,
    type=distutils.util.strtobool,
    help="Use sortagrad or not. (default: %(default)s)")
parser.add_argument(
    "--shuffle_method",
    default='instance_shuffle',
    type=str,
    help="Shuffle method: 'instance_shuffle', 'batch_shuffle', "
    "'batch_shuffle_batch'. (default: %(default)s)")
parser.add_argument(
    "--trainer_count",
    default=4,
    type=int,
    help="Trainer number. (default: %(default)s)")
parser.add_argument(
    "--mean_std_filepath",
    default='mean_std.npz',
    type=str,
    help="Manifest path for normalizer. (default: %(default)s)")
parser.add_argument(
    "--train_manifest_path",
    default='datasets/manifest.train',
    type=str,
    help="Manifest path for training. (default: %(default)s)")
parser.add_argument(
    "--dev_manifest_path",
    default='datasets/manifest.dev',
    type=str,
    help="Manifest path for validation. (default: %(default)s)")
parser.add_argument(
    "--vocab_filepath",
    default='datasets/vocab/eng_vocab.txt',
    type=str,
    help="Vocabulary filepath. (default: %(default)s)")
parser.add_argument(
    "--init_model_path",
    default=None,
    type=str,
    help="If set None, the training will start from scratch. "
    "Otherwise, the training will resume from "
    "the existing model of this path. (default: %(default)s)")
parser.add_argument(
    "--augmentation_config",
    default='{}',
    type=str,
    help="Augmentation configuration in json-format. "
    "(default: %(default)s)")
args = parser.parse_args()


def train():
    """DeepSpeech2 training."""

    # initialize data generator
    def data_generator():
        return DataGenerator(
            vocab_filepath=args.vocab_filepath,
            mean_std_filepath=args.mean_std_filepath,
            augmentation_config=args.augmentation_config)

    train_generator = data_generator()
    test_generator = data_generator()

    # create network config
    # paddle.data_type.dense_array is used for variable batch input.
    # The size 161 * 161 is only an placeholder value and the real shape
    # of input batch data will be induced during training.
    audio_data = paddle.layer.data(
        name="audio_spectrogram", type=paddle.data_type.dense_array(161 * 161))
    text_data = paddle.layer.data(
        name="transcript_text",
        type=paddle.data_type.integer_value_sequence(
            train_generator.vocab_size))
    cost = deep_speech2(
        audio_data=audio_data,
        text_data=text_data,
        dict_size=train_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_size=args.rnn_layer_size,
        is_inference=False)

    # create/load parameters and optimizer
    if args.init_model_path is None:
        parameters = paddle.parameters.create(cost)
    else:
        if not os.path.isfile(args.init_model_path):
            raise IOError("Invalid model!")
        parameters = paddle.parameters.Parameters.from_tar(
            gzip.open(args.init_model_path))
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.adam_learning_rate, gradient_clipping_threshold=400)
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # prepare data reader
    train_batch_reader = train_generator.batch_reader_creator(
        manifest_path=args.train_manifest_path,
        batch_size=args.batch_size,
        min_batch_size=args.trainer_count,
        sortagrad=args.use_sortagrad if args.init_model_path is None else False,
        shuffle_method=args.shuffle_method)
    test_batch_reader = test_generator.batch_reader_creator(
        manifest_path=args.dev_manifest_path,
        batch_size=args.batch_size,
        min_batch_size=1,  # must be 1, but will have errors.
        sortagrad=False,
        shuffle_method=None)

    # create event handler
    def event_handler(event):
        global start_time, cost_sum, cost_counter
        if isinstance(event, paddle.event.EndIteration):
            cost_sum += event.cost
            cost_counter += 1
            if (event.batch_id + 1) % 100 == 0:
                print("\nPass: %d, Batch: %d, TrainCost: %f" % (
                    event.pass_id, event.batch_id + 1, cost_sum / cost_counter))
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
            result = trainer.test(
                reader=test_batch_reader, feeding=test_generator.feeding)
            print("\n------- Time: %d sec,  Pass: %d, ValidationCost: %s" %
                  (time.time() - start_time, event.pass_id, result.cost))

    # run train
    trainer.train(
        reader=train_batch_reader,
        event_handler=event_handler,
        num_passes=args.num_passes,
        feeding=train_generator.feeding)


def main():
    utils.print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    train()


if __name__ == '__main__':
    main()
