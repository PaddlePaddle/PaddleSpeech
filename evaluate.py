"""Evaluation for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.v2 as paddle
import distutils.util
import argparse
import gzip
from data_utils.data import DataGenerator
from model import deep_speech2
from decoder import *
from scorer import Scorer
from error_rate import wer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--num_samples",
    default=100,
    type=int,
    help="Number of samples for evaluation. (default: %(default)s)")
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
    "--decode_method",
    default='beam_search_nproc',
    type=str,
    help="Method for ctc decoding, best_path, "
    "beam_search or beam_search_nproc. (default: %(default)s)")
parser.add_argument(
    "--language_model_path",
    default="data/en.00.UNKNOWN.klm",
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
parser.add_argument(
    "--beam_size",
    default=500,
    type=int,
    help="Width for beam search decoding. (default: %(default)d)")
parser.add_argument(
    "--normalizer_manifest_path",
    default='data/manifest.libri.train-clean-100',
    type=str,
    help="Manifest path for normalizer. (default: %(default)s)")
parser.add_argument(
    "--decode_manifest_path",
    default='data/manifest.libri.test-clean',
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
args = parser.parse_args()


def evaluate():
    """Evaluate on whole test data for DeepSpeech2."""

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

    # define inferer
    inferer = paddle.inference.Inference(
        output_layer=output_probs, parameters=parameters)

    # initialize external scorer for beam search decoding
    if args.decode_method == 'beam_search' or \
       args.decode_method == 'beam_search_nproc':
        ext_scorer = Scorer(args.alpha, args.beta, args.language_model_path)

    wer_counter, wer_sum = 0, 0.0
    for infer_data in batch_reader():
        # run inference
        infer_results = inferer.infer(input=infer_data)
        num_steps = len(infer_results) // len(infer_data)
        probs_split = [
            infer_results[i * num_steps:(i + 1) * num_steps]
            for i in xrange(0, len(infer_data))
        ]

        # decode and print
        # best path decode
        if args.decode_method == "best_path":
            for i, probs in enumerate(probs_split):
                output_transcription = ctc_best_path_decode(
                    probs_seq=probs, vocabulary=data_generator.vocab_list)
                target_transcription = ''.join([
                    data_generator.vocab_list[index]
                    for index in infer_data[i][1]
                ])
                wer_sum += wer(target_transcription, output_transcription)
                wer_counter += 1
        # beam search decode in single process
        elif args.decode_method == "beam_search":
            for i, probs in enumerate(probs_split):
                target_transcription = ''.join([
                    data_generator.vocab_list[index]
                    for index in infer_data[i][1]
                ])
                beam_search_result = ctc_beam_search_decoder(
                    probs_seq=probs,
                    vocabulary=data_generator.vocab_list,
                    beam_size=args.beam_size,
                    blank_id=len(data_generator.vocab_list),
                    ext_scoring_func=ext_scorer,
                    cutoff_prob=args.cutoff_prob, )
                wer_sum += wer(target_transcription, beam_search_result[0][1])
                wer_counter += 1
        # beam search using multiple processes
        elif args.decode_method == "beam_search_nproc":
            beam_search_nproc_results = ctc_beam_search_decoder_nproc(
                probs_split=probs_split,
                vocabulary=data_generator.vocab_list,
                beam_size=args.beam_size,
                blank_id=len(data_generator.vocab_list),
                ext_scoring_func=ext_scorer,
                cutoff_prob=args.cutoff_prob, )
            for i, beam_search_result in enumerate(beam_search_nproc_results):
                target_transcription = ''.join([
                    data_generator.vocab_list[index]
                    for index in infer_data[i][1]
                ])
                wer_sum += wer(target_transcription, beam_search_result[0][1])
                wer_counter += 1
        else:
            raise ValueError("Decoding method [%s] is not supported." %
                             decode_method)

        print("Cur WER = %f" % (wer_sum / wer_counter))
    print("Final WER = %f" % (wer_sum / wer_counter))


def main():
    paddle.init(use_gpu=args.use_gpu, trainer_count=1)
    evaluate()


if __name__ == '__main__':
    main()
