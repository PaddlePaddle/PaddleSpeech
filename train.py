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

NUM_CPU = multiprocessing.cpu_count() // 2
parser = argparse.ArgumentParser(description=__doc__)


def add_arg(argname, type, default, help, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    parser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


# yapf: disable
add_arg('batch_size',       int,    256,    "Minibatch size.")
add_arg('trainer_count',    int,    8,      "# of Trainers (CPUs or GPUs).")
add_arg('num_passes',       int,    200,    "# of training epochs.")
add_arg('parallels_data',   int,    NUM_CPU,"# of CPUs for data preprocessing.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('num_iter_print',   int,    100,    "Every # iterations for printing "
                                            "train cost.")
add_arg('learning_rate',    float,  5e-4,   "Learning rate.")
add_arg('max_duration',     float,  27.0,   "Longest audio duration allowed.")
add_arg('min_duration',     float,  0.0,    "Shortest audio duration allowed.")
add_arg('use_sortagrad',    bool,   True,   "Use SortaGrad or not.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('is_local',         bool,   True,   "Use pserver or not.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of Simple RNNs.")
add_arg('share_rnn_weights',bool,   True,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('train_manifest',   str,
        'datasets/manifest.train',
        "Filepath of train manifest.")
add_arg('dev_manifest',     str,
        'datasets/manifest.dev',
        "Filepath of validation manifest.")
add_arg('mean_std_path',    str,
        'mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'datasets/vocab/eng_vocab.txt',
        "Filepath of vocabulary.")
add_arg('init_model_path',  str,
        None,
        "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('output_model_dir', str,
        "./checkpoints",
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
args = parser.parse_args()
# yapf: disable


def train():
    """DeepSpeech2 training."""
    train_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config=open(args.augment_conf_path, 'r').read(),
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        specgram_type=args.specgram_type,
        num_threads=args.parallels_data)
    dev_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config="{}",
        specgram_type=args.specgram_type,
        num_threads=args.parallels_data)
    train_batch_reader = train_generator.batch_reader_creator(
        manifest_path=args.train_manifest,
        batch_size=args.batch_size,
        min_batch_size=args.trainer_count,
        sortagrad=args.use_sortagrad if args.init_model_path is None else False,
        shuffle_method=args.shuffle_method)
    dev_batch_reader = dev_generator.batch_reader_creator(
        manifest_path=args.dev_manifest,
        batch_size=args.batch_size,
        min_batch_size=1,  # must be 1, but will have errors.
        sortagrad=False,
        shuffle_method=None)

    ds2_model = DeepSpeech2Model(
        vocab_size=train_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        pretrained_model_path=args.init_model_path,
        share_rnn_weights=args.share_weights)
    ds2_model.train(
        train_batch_reader=train_batch_reader,
        dev_batch_reader=dev_batch_reader,
        feeding_dict=train_generator.feeding,
        learning_rate=args.learning_rate,
        gradient_clipping=400,
        num_passes=args.num_passes,
        num_iterations_print=args.num_iter_print,
        output_model_dir=args.output_model_dir,
        is_local=args.is_local)


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).iteritems()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def main():
    print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    train()


if __name__ == '__main__':
    main()
