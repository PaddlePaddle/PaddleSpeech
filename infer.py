"""Inferer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import distutils.util
import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model import deep_speech2
from decoder import *
from error_rate import wer
import utils

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--num_samples",
    default=100,
    type=int,
    help="Number of samples for inference. (default: %(default)s)")
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
    "--mean_std_filepath",
    default='mean_std.npz',
    type=str,
    help="Manifest path for normalizer. (default: %(default)s)")
parser.add_argument(
    "--decode_manifest_path",
    default='data/manifest.libri.test-100sample',
    type=str,
    help="Manifest path for decoding. (default: %(default)s)")
parser.add_argument(
    "--model_filepath",
    default='./params.tar.gz',
    type=str,
    help="Model filepath. (default: %(default)s)")
parser.add_argument(
    "--vocab_filepath",
    default='datasets/vocab/eng_vocab.txt',
    type=str,
    help="Vocabulary filepath. (default: %(default)s)")
parser.add_argument(
    "--decode_method",
    default='best_path',
    type=str,
    help="Method for ctc decoding:"
    "  best_path,"
    "  beam_search, "
    "   or beam_search_nproc. (default: %(default)s)")
parser.add_argument(
    "--beam_size",
    default=500,
    type=int,
    help="Width for beam search decoding. (default: %(default)d)")
parser.add_argument(
    "--num_results_per_sample",
    default=1,
    type=int,
    help="Number of output per sample in beam search. (default: %(default)d)")
parser.add_argument(
    "--language_model_path",
    default="data/1Billion.klm",
    type=str,
    help="Path for language model. (default: %(default)s)")
parser.add_argument(
    "--alpha",
    default=0.26,
    type=float,
    help="Parameter associated with language model. (default: %(default)f)")
parser.add_argument(
    "--beta",
    default=0.1,
    type=float,
    help="Parameter associated with word count. (default: %(default)f)")
parser.add_argument(
    "--cutoff_prob",
    default=0.99,
    type=float,
    help="The cutoff probability of pruning"
    "in beam search. (default: %(default)f)")
args = parser.parse_args()


def infer():
    """Inference for DeepSpeech2."""
    # initialize data generator
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_filepath,
        mean_std_filepath=args.mean_std_filepath,
        augmentation_config='{}')

    # create network config
    # paddle.data_type.dense_array is used for variable batch input.
    # The size 161 * 161 is only an placeholder value and the real shape
    # of input batch data will be induced during training.
    audio_data = paddle.layer.data(
        name="audio_spectrogram", type=paddle.data_type.dense_array(161 * 161))
    text_data = paddle.layer.data(
        name="transcript_text",
        type=paddle.data_type.integer_value_sequence(data_generator.vocab_size))
    output_probs = deep_speech2(
        audio_data=audio_data,
        text_data=text_data,
        dict_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_size=args.rnn_layer_size,
        is_inference=True)

    # load parameters
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(args.model_filepath))

    # prepare infer data
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.decode_manifest_path,
        batch_size=args.num_samples,
        sortagrad=False,
        shuffle_method=None)
    infer_data = batch_reader().next()

    # run inference
    infer_results = paddle.infer(
        output_layer=output_probs, parameters=parameters, input=infer_data)
    num_steps = len(infer_results) // len(infer_data)
    probs_split = [
        infer_results[i * num_steps:(i + 1) * num_steps]
        for i in xrange(len(infer_data))
    ]

    ## decode and print
    # best path decode
    wer_sum, wer_counter = 0, 0
    if args.decode_method == "best_path":
        for i, probs in enumerate(probs_split):
            target_transcription = ''.join([
                data_generator.vocab_list[index] for index in infer_data[i][1]
            ])
            best_path_transcription = ctc_best_path_decode(
                probs_seq=probs, vocabulary=data_generator.vocab_list)
            print("\nTarget Transcription: %s\nOutput Transcription: %s" %
                  (target_transcription, best_path_transcription))
            wer_cur = wer(target_transcription, best_path_transcription)
            wer_sum += wer_cur
            wer_counter += 1
            print("cur wer = %f, average wer = %f" %
                  (wer_cur, wer_sum / wer_counter))
    # beam search decode
    elif args.decode_method == "beam_search":
        ext_scorer = Scorer(args.alpha, args.beta, args.language_model_path)
        for i, probs in enumerate(probs_split):
            target_transcription = ''.join([
                data_generator.vocab_list[index] for index in infer_data[i][1]
            ])
            beam_search_result = ctc_beam_search_decoder(
                probs_seq=probs,
                vocabulary=data_generator.vocab_list,
                beam_size=args.beam_size,
                blank_id=len(data_generator.vocab_list),
                cutoff_prob=args.cutoff_prob,
                ext_scoring_func=ext_scorer, )
            print("\nTarget Transcription:\t%s" % target_transcription)

            for index in xrange(args.num_results_per_sample):
                result = beam_search_result[index]
                #output: index, log prob, beam result
                print("Beam %d: %f \t%s" % (index, result[0], result[1]))
            wer_cur = wer(target_transcription, beam_search_result[0][1])
            wer_sum += wer_cur
            wer_counter += 1
            print("cur wer = %f , average wer = %f" %
                  (wer_cur, wer_sum / wer_counter))
    elif args.decode_method == "beam_search_nproc":
        ext_scorer = Scorer(args.alpha, args.beta, args.language_model_path)
        beam_search_nproc_results = ctc_beam_search_decoder_nproc(
            probs_split=probs_split,
            vocabulary=data_generator.vocab_list,
            beam_size=args.beam_size,
            blank_id=len(data_generator.vocab_list),
            cutoff_prob=args.cutoff_prob,
            ext_scoring_func=ext_scorer, )
        for i, beam_search_result in enumerate(beam_search_nproc_results):
            target_transcription = ''.join([
                data_generator.vocab_list[index] for index in infer_data[i][1]
            ])
            print("\nTarget Transcription:\t%s" % target_transcription)

            for index in xrange(args.num_results_per_sample):
                result = beam_search_result[index]
                #output: index, log prob, beam result
                print("Beam %d: %f \t%s" % (index, result[0], result[1]))
            wer_cur = wer(target_transcription, beam_search_result[0][1])
            wer_sum += wer_cur
            wer_counter += 1
            print("cur wer = %f , average wer = %f" %
                  (wer_cur, wer_sum / wer_counter))
    else:
        raise ValueError("Decoding method [%s] is not supported." %
                         decode_method)


def main():
    utils.print_arguments(args)
    paddle.init(use_gpu=args.use_gpu, trainer_count=1)
    infer()


if __name__ == '__main__':
    main()
