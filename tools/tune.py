"""Beam search parameters tuning for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import argparse
import functools
import gzip
import logging
import paddle.v2 as paddle
import _init_paths
from data_utils.data import DataGenerator
from decoders.swig_wrapper import Scorer
from decoders.swig_wrapper import ctc_beam_search_decoder_batch
from model_utils.model import deep_speech_v2_network
from utils.error_rate import wer, cer
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('num_batches',      int,    -1,     "# of batches tuning on. "
                                            "Default -1, on whole dev set.")
add_arg('batch_size',       int,    256,    "# of samples per batch.")
add_arg('trainer_count',    int,    8,      "# of Trainers (CPUs or GPUs).")
add_arg('beam_size',        int,    500,    "Beam search width.")
add_arg('num_proc_bsearch', int,    12,     "# of CPUs for beam search.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('num_alphas',       int,    45,     "# of alpha candidates for tuning.")
add_arg('num_betas',        int,    8,      "# of beta candidates for tuning.")
add_arg('alpha_from',       float,  1.0,    "Where alpha starts tuning from.")
add_arg('alpha_to',         float,  3.2,    "Where alpha ends tuning with.")
add_arg('beta_from',        float,  0.1,    "Where beta starts tuning from.")
add_arg('beta_to',          float,  0.45,   "Where beta ends tuning with.")
add_arg('cutoff_prob',      float,  1.0,    "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   True,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('tune_manifest',    str,
        'data/librispeech/manifest.dev-clean',
        "Filepath of manifest to tune.")
add_arg('mean_std_path',    str,
        'data/librispeech/mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'data/librispeech/vocab.txt',
        "Filepath of vocabulary.")
add_arg('lang_model_path',  str,
        'models/lm/common_crawl_00.prune01111.trie.klm',
        "Filepath for language model.")
add_arg('model_path',       str,
        './checkpoints/libri/params.latest.tar.gz',
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
# yapf: disable
args = parser.parse_args()


logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

def tune():
    """Tune parameters alpha and beta incrementally."""
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

    audio_data = paddle.layer.data(
        name="audio_spectrogram",
        type=paddle.data_type.dense_array(161 * 161))
    text_data = paddle.layer.data(
        name="transcript_text",
        type=paddle.data_type.integer_value_sequence(data_generator.vocab_size))

    output_probs, _ = deep_speech_v2_network(
        audio_data=audio_data,
        text_data=text_data,
        dict_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        share_rnn_weights=args.share_rnn_weights)

    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.tune_manifest,
        batch_size=args.batch_size,
        sortagrad=False,
        shuffle_method=None)

    # load parameters
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(args.model_path))

    inferer = paddle.inference.Inference(
        output_layer=output_probs, parameters=parameters)
    # decoders only accept string encoded in utf-8
    vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]

    # init logger
    logger = logging.getLogger("")
    logger.setLevel(level=logging.INFO)
    # init external scorer
    logger.info("begin to initialize the external scorer for tuning")
    ext_scorer = Scorer(
        alpha=args.alpha_from,
        beta=args.beta_from,
        model_path=args.lang_model_path,
        vocabulary=vocab_list)
    logger.info("language model: "
                "is_character_based = %d," % ext_scorer.is_character_based() +
                " max_order = %d," % ext_scorer.get_max_order() +
                " dict_size = %d" % ext_scorer.get_dict_size())
    logger.info("end initializing scorer. Start tuning ...")

    error_rate_func = cer if args.error_rate_type == 'cer' else wer
    # create grid for search
    cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
    cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
    params_grid = [(alpha, beta) for alpha in cand_alphas
                   for beta in cand_betas]

    err_sum = [0.0 for i in xrange(len(params_grid))]
    err_ave = [0.0 for i in xrange(len(params_grid))]
    num_ins, cur_batch = 0, 0
    ## incremental tuning parameters over multiple batches
    for infer_data in batch_reader():
        if (args.num_batches >= 0) and (cur_batch >= args.num_batches):
            break
        infer_results = inferer.infer(input=infer_data)

        num_steps = len(infer_results) // len(infer_data)
        probs_split = [
            infer_results[i * num_steps:(i + 1) * num_steps]
            for i in xrange(len(infer_data))
        ]

        target_transcripts = [
            ''.join([data_generator.vocab_list[token] for token in transcript])
            for _, transcript in infer_data
        ]

        num_ins += len(target_transcripts)
        # grid search
        for index, (alpha, beta) in enumerate(params_grid):
            # reset alpha & beta
            ext_scorer.reset_params(alpha, beta)
            beam_search_results = ctc_beam_search_decoder_batch(
                probs_split=probs_split,
                vocabulary=vocab_list,
                beam_size=args.beam_size,
                num_processes=args.num_proc_bsearch,
                cutoff_prob=args.cutoff_prob,
                cutoff_top_n=args.cutoff_top_n,
                ext_scoring_func=ext_scorer, )

            result_transcripts = [res[0][1] for res in beam_search_results]
            for target, result in zip(target_transcripts, result_transcripts):
                err_sum[index] += error_rate_func(target, result)
            err_ave[index] = err_sum[index] / num_ins
            if index % 2 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

        # output on-line tuning result at the end of current batch
        err_ave_min = min(err_ave)
        min_index = err_ave.index(err_ave_min)
        print("\nBatch %d [%d/?], current opt (alpha, beta) = (%s, %s), "
              " min [%s] = %f" %(cur_batch, num_ins,
              "%.3f" % params_grid[min_index][0],
              "%.3f" % params_grid[min_index][1],
              args.error_rate_type, err_ave_min))
        cur_batch += 1

    # output WER/CER at every (alpha, beta)
    print("\nFinal %s:\n" % args.error_rate_type)
    for index in xrange(len(params_grid)):
        print("(alpha, beta) = (%s, %s), [%s] = %f"
             % ("%.3f" % params_grid[index][0], "%.3f" % params_grid[index][1],
             args.error_rate_type, err_ave[index]))

    err_ave_min = min(err_ave)
    min_index = err_ave.index(err_ave_min)
    print("\nFinish tuning on %d batches, final opt (alpha, beta) = (%s, %s)"
            % (args.num_batches, "%.3f" % params_grid[min_index][0],
              "%.3f" % params_grid[min_index][1]))

    logger.info("finish tuning")


def main():
    print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    tune()


if __name__ == '__main__':
    main()
