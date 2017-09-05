"""Beam search parameters tuning for DeepSpeech2 model."""
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
add_arg('num_samples',      int,    100,    "# of samples to infer.")
add_arg('trainer_count',    int,    8,      "# of Trainers (CPUs or GPUs).")
add_arg('beam_size',        int,    500,    "Beam search width.")
add_arg('parallels_bsearch',int,    NUM_CPU,"# of CPUs for beam search.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('num_alphas',       int,    14,     "# of alpha candidates for tuning.")
add_arg('num_betas',        int,    20,     "# of beta candidates for tuning.")
add_arg('alpha_from',       float,  0.1,    "Where alpha starts tuning from.")
add_arg('alpha_to',         float,  0.36,   "Where alpha ends tuning with.")
add_arg('beta_from',        float,  0.05,   "Where beta starts tuning from.")
add_arg('beta_to',          float,  0.36,   "Where beta ends tuning with.")
add_arg('cutoff_prob',      float,  0.99,   "Cutoff probability for pruning.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of Simple RNNs.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   True,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('tune_manifest',    str,
        'datasets/manifest.test',
        "Filepath of manifest to tune.")
add_arg('mean_std_path',    str,
        'mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'datasets/vocab/eng_vocab.txt',
        "Filepath of vocabulary.")
add_arg('lang_model_path',  str,
        'lm/data/common_crawl_00.prune01111.trie.klm',
        "Filepath for language model.")
add_arg('model_path',       str,
        './checkpoints/params.latest.tar.gz',
        "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('error_rate_type',  str,
        'wer',
        "Error rate type for evaluation.",
        choices=['wer', 'cer'])
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
args = parser.parse_args()
# yapf: disable


def tune():
    """Tune parameters alpha and beta on one minibatch."""
    if not args.num_alphas >= 0:
        raise ValueError("num_alphas must be non-negative!")
    if not args.num_betas >= 0:
        raise ValueError("num_betas must be non-negative!")

    data_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        num_threads=1)
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.tune_manifest,
        batch_size=args.num_samples,
        sortagrad=False,
        shuffle_method=None)
    tune_data = batch_reader().next()
    target_transcripts = [
        ''.join([data_generator.vocab_list[token] for token in transcript])
        for _, transcript in tune_data
    ]

    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        pretrained_model_path=args.model_path,
        share_rnn_weights=args.share_rnn_weights)

    # create grid for search
    cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
    cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
    params_grid = [(alpha, beta) for alpha in cand_alphas
                   for beta in cand_betas]

    ## tune parameters in loop
    for alpha, beta in params_grid:
        result_transcripts = ds2_model.infer_batch(
            infer_data=tune_data,
            decoder_method='ctc_beam_search',
            beam_alpha=alpha,
            beam_beta=beta,
            beam_size=args.beam_size,
            cutoff_prob=args.cutoff_prob,
            vocab_list=data_generator.vocab_list,
            language_model_path=args.lang_model_path,
            num_processes=args.parallels_bsearch)
        wer_sum, num_ins = 0.0, 0
        for target, result in zip(target_transcripts, result_transcripts):
            wer_sum += wer(target, result)
            num_ins += 1
        print("alpha = %f\tbeta = %f\tWER = %f" %
              (alpha, beta, wer_sum / num_ins))


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).iteritems()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def main():
    print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    tune()


if __name__ == '__main__':
    main()
