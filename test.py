"""Evaluation for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.error_rate import wer, cer
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,    128,    "Minibatch size.")
add_arg('trainer_count',    int,    8,      "# of Trainers (CPUs or GPUs).")
add_arg('beam_size',        int,    500,    "Beam search width.")
add_arg('num_proc_bsearch', int,    8,      "# of CPUs for beam search.")
add_arg('num_proc_data',    int,    8,      "# of CPUs for data preprocessing.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    2048,   "# of recurrent cells per layer.")
add_arg('alpha',            float,  2.15,   "Coef of LM for beam search.")
add_arg('beta',             float,  0.35,   "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  1.0,    "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
add_arg('use_gru',          bool,   False,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   True,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   True,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('test_manifest',   str,
        'data/librispeech/manifest.test-clean',
        "Filepath of manifest to evaluate.")
add_arg('mean_std_path',    str,
        'data/librispeech/mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'data/librispeech/vocab.txt',
        "Filepath of vocabulary.")
add_arg('model_path',       str,
        './checkpoints/libri/params.latest.tar.gz',
        "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('lang_model_path',  str,
        'models/lm/common_crawl_00.prune01111.trie.klm',
        "Filepath for language model.")
add_arg('decoding_method',  str,
        'ctc_beam_search',
        "Decoding method. Options: ctc_beam_search, ctc_greedy",
        choices = ['ctc_beam_search', 'ctc_greedy'])
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


def evaluate():
    """Evaluate on whole test data for DeepSpeech2."""
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        num_threads=args.num_proc_data,
        keep_transcription_text=True)
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.test_manifest,
        batch_size=args.batch_size,
        min_batch_size=1,
        sortagrad=False,
        shuffle_method=None)

    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        pretrained_model_path=args.model_path,
        share_rnn_weights=args.share_rnn_weights)

    # decoders only accept string encoded in utf-8
    vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]

    error_rate_func = cer if args.error_rate_type == 'cer' else wer
    error_sum, num_ins = 0.0, 0
    for infer_data in batch_reader():
        result_transcripts = ds2_model.infer_batch(
            infer_data=infer_data,
            decoding_method=args.decoding_method,
            beam_alpha=args.alpha,
            beam_beta=args.beta,
            beam_size=args.beam_size,
            cutoff_prob=args.cutoff_prob,
            cutoff_top_n=args.cutoff_top_n,
            vocab_list=vocab_list,
            language_model_path=args.lang_model_path,
            num_processes=args.num_proc_bsearch)
        target_transcripts = [
            transcript for _, transcript in infer_data
        ]
        for target, result in zip(target_transcripts, result_transcripts):
            error_sum += error_rate_func(target, result)
            num_ins += 1
        print("Error rate [%s] (%d/?) = %f" %
              (args.error_rate_type, num_ins, error_sum / num_ins))
    print("Final error rate [%s] (%d/%d) = %f" %
          (args.error_rate_type, num_ins, num_ins, error_sum / num_ins))

    ds2_model.logger.info("finish evaluation")

def main():
    print_arguments(args)
    paddle.init(use_gpu=args.use_gpu,
                rnn_use_batch=True,
                trainer_count=args.trainer_count)
    evaluate()


if __name__ == '__main__':
    main()
