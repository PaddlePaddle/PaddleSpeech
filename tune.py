"""Parameters tuning for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import distutils.util
import argparse
import multiprocessing
import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model import DeepSpeech2Model
from error_rate import wer
import utils

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--batch_size",
    default=128,
    type=int,
    help="Minibatch size for parameters tuning. (default: %(default)s)")
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
    "--trainer_count",
    default=8,
    type=int,
    help="Trainer number. (default: %(default)s)")
parser.add_argument(
    "--num_threads_data",
    default=1,
    type=int,
    help="Number of cpu threads for preprocessing data. (default: %(default)s)")
parser.add_argument(
    "--num_processes_beam_search",
    default=multiprocessing.cpu_count(),
    type=int,
    help="Number of cpu processes for beam search. (default: %(default)s)")
parser.add_argument(
    "--specgram_type",
    default='linear',
    type=str,
    help="Feature type of audio data: 'linear' (power spectrum)"
    " or 'mfcc'. (default: %(default)s)")
parser.add_argument(
    "--mean_std_filepath",
    default='mean_std.npz',
    type=str,
    help="Manifest path for normalizer. (default: %(default)s)")
parser.add_argument(
    "--tune_manifest_path",
    default='datasets/manifest.dev',
    type=str,
    help="Manifest path for tuning. (default: %(default)s)")
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
    "--beam_size",
    default=500,
    type=int,
    help="Width for beam search decoding. (default: %(default)d)")
parser.add_argument(
    "--language_model_path",
    default="lm/data/common_crawl_00.prune01111.trie.klm",
    type=str,
    help="Path for language model. (default: %(default)s)")
parser.add_argument(
    "--alpha_from",
    default=0.1,
    type=float,
    help="Where alpha starts from. (default: %(default)f)")
parser.add_argument(
    "--num_alphas",
    default=14,
    type=int,
    help="Number of candidate alphas. (default: %(default)d)")
parser.add_argument(
    "--alpha_to",
    default=0.36,
    type=float,
    help="Where alpha ends with. (default: %(default)f)")
parser.add_argument(
    "--beta_from",
    default=0.05,
    type=float,
    help="Where beta starts from. (default: %(default)f)")
parser.add_argument(
    "--num_betas",
    default=20,
    type=float,
    help="Number of candidate betas. (default: %(default)d)")
parser.add_argument(
    "--beta_to",
    default=1.0,
    type=float,
    help="Where beta ends with. (default: %(default)f)")
parser.add_argument(
    "--cutoff_prob",
    default=0.99,
    type=float,
    help="The cutoff probability of pruning"
    "in beam search. (default: %(default)f)")
args = parser.parse_args()


def tune():
    """Tune parameters alpha and beta for the CTC beam search decoder
    incrementally. The optimal parameters up to now would be output real time
    at the end of each minibatch data, until all the development data is
    taken into account. And the tuning process can be terminated at any time
    as long as the two parameters get stable.
    """
    if not args.num_alphas >= 0:
        raise ValueError("num_alphas must be non-negative!")
    if not args.num_betas >= 0:
        raise ValueError("num_betas must be non-negative!")

    data_generator = DataGenerator(
        vocab_filepath=args.vocab_filepath,
        mean_std_filepath=args.mean_std_filepath,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        num_threads=args.num_threads_data)
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.tune_manifest_path,
        batch_size=args.batch_size,
        sortagrad=False,
        shuffle_method=None)

    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        pretrained_model_path=args.model_filepath)

    # create grid for search
    cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
    cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
    params_grid = [(alpha, beta) for alpha in cand_alphas
                   for beta in cand_betas]

    wer_sum = [0.0 for i in xrange(len(params_grid))]
    ave_wer = [0.0 for i in xrange(len(params_grid))]
    num_ins = 0
    num_batches = 0
    ## incremental tuning parameters over multiple batches
    for infer_data in batch_reader():
        target_transcripts = [
            ''.join([data_generator.vocab_list[token] for token in transcript])
            for _, transcript in infer_data
        ]

        num_ins += len(target_transcripts)
        # grid search
        for index, (alpha, beta) in enumerate(params_grid):
            result_transcripts = ds2_model.infer_batch(
                infer_data=infer_data,
                decode_method='beam_search',
                beam_alpha=alpha,
                beam_beta=beta,
                beam_size=args.beam_size,
                cutoff_prob=args.cutoff_prob,
                vocab_list=data_generator.vocab_list,
                language_model_path=args.language_model_path,
                num_processes=args.num_processes_beam_search)

            for target, result in zip(target_transcripts, result_transcripts):
                wer_sum[index] += wer(target, result)
            ave_wer[index] = wer_sum[index] / num_ins
            print("alpha = %f, beta = %f, WER = %f" %
                  (alpha, beta, ave_wer[index]))

        # output on-line tuning result at the the end of current batch
        ave_wer_min = min(ave_wer)
        min_index = ave_wer.index(ave_wer_min)
        print("Finish batch %d, optimal (alpha, beta, WER) = (%f, %f, %f)\n" %
              (num_batches, params_grid[min_index][0],
               params_grid[min_index][1], ave_wer_min))
        num_batches += 1


def main():
    utils.print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    tune()


if __name__ == '__main__':
    main()
