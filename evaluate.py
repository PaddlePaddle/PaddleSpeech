"""Evaluation for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils.util
import argparse
import multiprocessing
import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model import DeepSpeech2Model
from error_rate import wer
from error_rate import cer
import utils

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--batch_size",
    default=128,
    type=int,
    help="Minibatch size for evaluation. (default: %(default)s)")
parser.add_argument(
    "--trainer_count",
    default=8,
    type=int,
    help="Trainer number. (default: %(default)s)")
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
    default=multiprocessing.cpu_count() // 2,
    type=int,
    help="Number of cpu threads for preprocessing data. (default: %(default)s)")
parser.add_argument(
    "--num_processes_beam_search",
    default=multiprocessing.cpu_count() // 2,
    type=int,
    help="Number of cpu processes for beam search. (default: %(default)s)")
parser.add_argument(
    "--mean_std_filepath",
    default='mean_std.npz',
    type=str,
    help="Manifest path for normalizer. (default: %(default)s)")
parser.add_argument(
    "--decode_method",
    default='beam_search',
    type=str,
    help="Method for ctc decoding, best_path or beam_search. "
    "(default: %(default)s)")
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
parser.add_argument(
    "--beam_size",
    default=500,
    type=int,
    help="Width for beam search decoding. (default: %(default)d)")
parser.add_argument(
    "--specgram_type",
    default='linear',
    type=str,
    help="Feature type of audio data: 'linear' (power spectrum)"
    " or 'mfcc'. (default: %(default)s)")
parser.add_argument(
    "--decode_manifest_path",
    default='datasets/manifest.test',
    type=str,
    help="Manifest path for decoding. (default: %(default)s)")
parser.add_argument(
    "--model_filepath",
    default='checkpoints/params.latest.tar.gz',
    type=str,
    help="Model filepath. (default: %(default)s)")
parser.add_argument(
    "--vocab_filepath",
    default='datasets/vocab/eng_vocab.txt',
    type=str,
    help="Vocabulary filepath. (default: %(default)s)")
parser.add_argument(
    "--error_rate_type",
    default='wer',
    choices=['wer', 'cer'],
    type=str,
    help="There are total two error rate types including wer and cer. wer "
    "represents for word error rate while cer for character error rate. "
    "(default: %(default)s)")
args = parser.parse_args()


def evaluate():
    """Evaluate on whole test data for DeepSpeech2."""
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_filepath,
        mean_std_filepath=args.mean_std_filepath,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        num_threads=args.num_threads_data)
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.decode_manifest_path,
        batch_size=args.batch_size,
        min_batch_size=1,
        sortagrad=False,
        shuffle_method=None)

    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        pretrained_model_path=args.model_filepath)

    if args.error_rate_type == 'wer':
        error_rate_func = wer
        error_rate_info = 'WER'
    else:
        error_rate_func = cer
        error_rate_info = 'CER'

    error_sum, num_ins = 0.0, 0
    for infer_data in batch_reader():
        result_transcripts = ds2_model.infer_batch(
            infer_data=infer_data,
            decode_method=args.decode_method,
            beam_alpha=args.alpha,
            beam_beta=args.beta,
            beam_size=args.beam_size,
            cutoff_prob=args.cutoff_prob,
            vocab_list=data_generator.vocab_list,
            language_model_path=args.language_model_path,
            num_processes=args.num_processes_beam_search)
        target_transcripts = [
            ''.join([data_generator.vocab_list[token] for token in transcript])
            for _, transcript in infer_data
        ]
        for target, result in zip(target_transcripts, result_transcripts):
            error_sum += error_rate_func(target, result)
            num_ins += 1
        print("%s (%d/?) = %f" % \
                (error_rate_info, num_ins, error_sum / num_ins))
    print("Final %s (%d/%d) = %f" % \
            (error_rate_info, num_ins, num_ins, error_sum / num_ins))


def main():
    utils.print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    evaluate()


if __name__ == '__main__':
    main()
