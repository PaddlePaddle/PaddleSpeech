"""Trainer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import multiprocessing
import paddle.v2 as paddle
from model import DeepSpeech2Model
from data_utils.data import DataGenerator
import utils

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--batch_size", default=256, type=int, help="Minibatch size.")
parser.add_argument(
    "--num_passes",
    default=200,
    type=int,
    help="Training pass number. (default: %(default)s)")
parser.add_argument(
    "--num_iterations_print",
    default=100,
    type=int,
    help="Number of iterations for every train cost printing. "
    "(default: %(default)s)")
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
    "--specgram_type",
    default='linear',
    type=str,
    help="Feature type of audio data: 'linear' (power spectrum)"
    " or 'mfcc'. (default: %(default)s)")
parser.add_argument(
    "--max_duration",
    default=27.0,
    type=float,
    help="Audios with duration larger than this will be discarded. "
    "(default: %(default)s)")
parser.add_argument(
    "--min_duration",
    default=0.0,
    type=float,
    help="Audios with duration smaller than this will be discarded. "
    "(default: %(default)s)")
parser.add_argument(
    "--shuffle_method",
    default='batch_shuffle_clipped',
    type=str,
    help="Shuffle method: 'instance_shuffle', 'batch_shuffle', "
    "'batch_shuffle_batch'. (default: %(default)s)")
parser.add_argument(
    "--trainer_count",
    default=8,
    type=int,
    help="Trainer number. (default: %(default)s)")
parser.add_argument(
    "--num_threads_data",
    default=multiprocessing.cpu_count() // 2,
    type=int,
    help="Number of cpu threads for preprocessing data. (default: %(default)s)")
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
    "--output_model_dir",
    default="./checkpoints",
    type=str,
    help="Directory for saving models. (default: %(default)s)")
parser.add_argument(
    "--augmentation_config",
    default=open('augmentation.config', 'r').read(),
    type=str,
    help="Augmentation configuration in json-format. "
    "(default: %(default)s)")
args = parser.parse_args()


def train():
    """DeepSpeech2 training."""
    train_generator = DataGenerator(
        vocab_filepath=args.vocab_filepath,
        mean_std_filepath=args.mean_std_filepath,
        augmentation_config=args.augmentation_config,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        specgram_type=args.specgram_type,
        num_threads=args.num_threads_data)
    dev_generator = DataGenerator(
        vocab_filepath=args.vocab_filepath,
        mean_std_filepath=args.mean_std_filepath,
        augmentation_config="{}",
        specgram_type=args.specgram_type,
        num_threads=args.num_threads_data)
    train_batch_reader = train_generator.batch_reader_creator(
        manifest_path=args.train_manifest_path,
        batch_size=args.batch_size,
        min_batch_size=args.trainer_count,
        sortagrad=args.use_sortagrad if args.init_model_path is None else False,
        shuffle_method=args.shuffle_method)
    dev_batch_reader = dev_generator.batch_reader_creator(
        manifest_path=args.dev_manifest_path,
        batch_size=args.batch_size,
        min_batch_size=1,  # must be 1, but will have errors.
        sortagrad=False,
        shuffle_method=None)

    ds2_model = DeepSpeech2Model(
        vocab_size=train_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        pretrained_model_path=args.init_model_path)
    ds2_model.train(
        train_batch_reader=train_batch_reader,
        dev_batch_reader=dev_batch_reader,
        feeding_dict=train_generator.feeding,
        learning_rate=args.adam_learning_rate,
        gradient_clipping=400,
        num_passes=args.num_passes,
        num_iterations_print=args.num_iterations_print,
        output_model_dir=args.output_model_dir)


def main():
    utils.print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    train()


if __name__ == '__main__':
    main()
