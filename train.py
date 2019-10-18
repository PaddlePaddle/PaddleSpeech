"""Trainer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import io
from model_utils.model import DeepSpeech2Model
from model_utils.model_check import check_cuda, check_version
from data_utils.data import DataGenerator
from utils.utility import add_arguments, print_arguments

import paddle.fluid as fluid

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,    256,    "Minibatch size.")
add_arg('num_epoch',       int,    200,    "# of training epochs.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('num_iter_print',   int,    100,    "Every # batch for printing "
                                            "train cost.")
add_arg('save_epoch',   int,    10,   "# Every # batch for save checkpoint and modle params ")
add_arg('num_samples',    int,    10000,    "The num of train samples.")
add_arg('learning_rate',    float,  5e-4,   "Learning rate.")
add_arg('max_duration',     float,  27.0,   "Longest audio duration allowed.")
add_arg('min_duration',     float,  0.0,    "Shortest audio duration allowed.")
add_arg('test_off',         bool,   False,  "Turn off testing.")
add_arg('use_sortagrad',    bool,   True,   "Use SortaGrad or not.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of simple RNNs.")
add_arg('is_local',         bool,   True,   "Use pserver or not.")
add_arg('share_rnn_weights',bool,   True,   "Share input-hidden weights across "
                                           "bi-directional RNNs. Not for GRU.")
add_arg('init_from_pretrained_model',str,
         None,
         "If None, the training starts from scratch, "
         "otherwise, it resumes from the pre-trained model.")

add_arg('train_manifest',   str,
        'data/librispeech/manifest.train',
        "Filepath of train manifest.")
add_arg('dev_manifest',     str,
        'data/librispeech/manifest.dev-clean',
        "Filepath of validation manifest.")
add_arg('mean_std_path',    str,
        'data/librispeech/mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'data/librispeech/vocab.txt',
        "Filepath of vocabulary.")
add_arg('output_model_dir', str,
        "./checkpoints/libri",
        "Directory for saving checkpoints.")
add_arg('augment_conf_path',str,
        'conf/augmentation.config',
        "Filepath of augmentation configuration file (json-format).")
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
add_arg('shuffle_method',   str,
        'batch_shuffle_clipped',
        "Shuffle method.",
        choices=['instance_shuffle', 'batch_shuffle', 'batch_shuffle_clipped'])
# yapf: disable
args = parser.parse_args()


def train():
    """DeepSpeech2 training."""

    # check if set use_gpu=True in paddlepaddle cpu version
    check_cuda(args.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    train_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config=io.open(args.augment_conf_path, mode='r', encoding='utf8').read(),
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        specgram_type=args.specgram_type,
        place=place)
    dev_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config="{}",
        specgram_type=args.specgram_type,
        place = place)
    train_batch_reader = train_generator.batch_reader_creator(
        manifest_path=args.train_manifest,
        batch_size=args.batch_size,
        sortagrad=args.use_sortagrad if args.init_from_pretrained_model is None else False,
        shuffle_method=args.shuffle_method)
    dev_batch_reader = dev_generator.batch_reader_creator(
        manifest_path=args.dev_manifest,
        batch_size=args.batch_size,
        sortagrad=False,
        shuffle_method=None)

    ds2_model = DeepSpeech2Model(
        vocab_size=train_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        share_rnn_weights=args.share_rnn_weights,
        place=place,
        init_from_pretrained_model=args.init_from_pretrained_model,
        output_model_dir=args.output_model_dir)

    ds2_model.train(
        train_batch_reader=train_batch_reader,
        dev_batch_reader=dev_batch_reader,
        feeding_dict=train_generator.feeding,
        learning_rate=args.learning_rate,
        gradient_clipping=400,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_epoch=args.num_epoch,
        save_epoch=args.save_epoch,
        num_iterations_print=args.num_iter_print,
        test_off=args.test_off)


def main():
    print_arguments(args)
    train()


if __name__ == '__main__':
    main()
