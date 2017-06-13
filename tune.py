"""
   Tune parameters for beam search decoder in Deep Speech 2.
"""

import paddle.v2 as paddle
import distutils.util
import argparse
import gzip
from audio_data_utils import DataGenerator
from model import deep_speech2
from decoder import *
from error_rate import wer

parser = argparse.ArgumentParser(
    description='Parameters tuning script for ctc beam search decoder in  Deep Speech 2.'
)
parser.add_argument(
    "--num_samples",
    default=100,
    type=int,
    help="Number of samples for parameters tuning. (default: %(default)s)")
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
    "--normalizer_manifest_path",
    default='data/manifest.libri.train-clean-100',
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
    default='data/eng_vocab.txt',
    type=str,
    help="Vocabulary filepath. (default: %(default)s)")
parser.add_argument(
    "--decode_method",
    default='beam_search_nproc',
    type=str,
    help="Method for decoding, beam_search or beam_search_nproc. (default: %(default)s)"
)
parser.add_argument(
    "--beam_size",
    default=500,
    type=int,
    help="Width for beam search decoding. (default: %(default)d)")
parser.add_argument(
    "--num_results_per_sample",
    default=1,
    type=int,
    help="Number of outputs per sample in beam search. (default: %(default)d)")
parser.add_argument(
    "--language_model_path",
    default="./data/1Billion.klm",
    type=str,
    help="Path for language model. (default: %(default)s)")
parser.add_argument(
    "--alpha_from",
    default=0.0,
    type=float,
    help="Where alpha starts from, <= alpha_to. (default: %(default)f)")
parser.add_argument(
    "--alpha_stride",
    default=0.001,
    type=float,
    help="Step length for varying alpha. (default: %(default)f)")
parser.add_argument(
    "--alpha_to",
    default=0.01,
    type=float,
    help="Where alpha ends with, >= alpha_from. (default: %(default)f)")
parser.add_argument(
    "--beta_from",
    default=0.0,
    type=float,
    help="Where beta starts from, <= beta_to. (default: %(default)f)")
parser.add_argument(
    "--beta_stride",
    default=0.01,
    type=float,
    help="Step length for varying beta. (default: %(default)f)")
parser.add_argument(
    "--beta_to",
    default=0.0,
    type=float,
    help="Where beta ends with, >= beta_from. (default: %(default)f)")
args = parser.parse_args()


def tune():
    """
    Tune parameters alpha and beta on one minibatch.
    """

    if not args.alpha_from <= args.alpha_to:
        raise ValueError("alpha_from <= alpha_to doesn't satisfy!")
    if not args.alpha_stride > 0:
        raise ValueError("alpha_stride shouldn't be negative!")

    if not args.beta_from <= args.beta_to:
        raise ValueError("beta_from <= beta_to doesn't satisfy!")
    if not args.beta_stride > 0:
        raise ValueError("beta_stride shouldn't be negative!")

    # initialize data generator
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_filepath,
        normalizer_manifest_path=args.normalizer_manifest_path,
        normalizer_num_samples=200,
        max_duration=20.0,
        min_duration=0.0,
        stride_ms=10,
        window_ms=20)

    # create network config
    dict_size = data_generator.vocabulary_size()
    vocab_list = data_generator.vocabulary_list()
    audio_data = paddle.layer.data(
        name="audio_spectrogram",
        height=161,
        width=2000,
        type=paddle.data_type.dense_vector(322000))
    text_data = paddle.layer.data(
        name="transcript_text",
        type=paddle.data_type.integer_value_sequence(dict_size))
    output_probs = deep_speech2(
        audio_data=audio_data,
        text_data=text_data,
        dict_size=dict_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_size=args.rnn_layer_size,
        is_inference=True)

    # load parameters
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(args.model_filepath))

    # prepare infer data
    feeding = data_generator.data_name_feeding()
    test_batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.decode_manifest_path,
        batch_size=args.num_samples,
        padding_to=2000,
        flatten=True,
        sort_by_duration=False,
        shuffle=False)
    infer_data = test_batch_reader().next()

    # run inference
    infer_results = paddle.infer(
        output_layer=output_probs, parameters=parameters, input=infer_data)
    num_steps = len(infer_results) / len(infer_data)
    probs_split = [
        infer_results[i * num_steps:(i + 1) * num_steps]
        for i in xrange(0, len(infer_data))
    ]

    cand_alpha = np.arange(args.alpha_from, args.alpha_to + args.alpha_stride,
                           args.alpha_stride)
    cand_beta = np.arange(args.beta_from, args.beta_to + args.beta_stride,
                          args.beta_stride)
    params_grid = [(alpha, beta) for alpha in cand_alpha for beta in cand_beta]
    ## tune parameters in loop
    for (alpha, beta) in params_grid:
        wer_sum, wer_counter = 0, 0
        ext_scorer = Scorer(alpha, beta, args.language_model_path)
        # beam search decode
        if args.decode_method == "beam_search":
            for i, probs in enumerate(probs_split):
                target_transcription = ''.join(
                    [vocab_list[index] for index in infer_data[i][1]])
                beam_search_result = ctc_beam_search_decoder(
                    probs_seq=probs,
                    vocabulary=vocab_list,
                    beam_size=args.beam_size,
                    ext_scoring_func=ext_scorer,
                    blank_id=len(vocab_list))
                wer_sum += wer(target_transcription, beam_search_result[0][1])
                wer_counter += 1
        # beam search using multiple processes
        elif args.decode_method == "beam_search_nproc":
            beam_search_nproc_results = ctc_beam_search_decoder_nproc(
                probs_split=probs_split,
                vocabulary=vocab_list,
                beam_size=args.beam_size,
                ext_scoring_func=ext_scorer,
                blank_id=len(vocab_list),
                num_processes=1)
            for i, beam_search_result in enumerate(beam_search_nproc_results):
                target_transcription = ''.join(
                    [vocab_list[index] for index in infer_data[i][1]])
                wer_sum += wer(target_transcription, beam_search_result[0][1])
                wer_counter += 1
        else:
            raise ValueError("Decoding method [%s] is not supported." % method)

        print("alpha = %f\tbeta = %f\tWER = %f" %
              (alpha, beta, wer_sum / wer_counter))


def main():
    paddle.init(use_gpu=args.use_gpu, trainer_count=1)
    tune()


if __name__ == '__main__':
    main()
