"""
   Inference for a simplifed version of Baidu DeepSpeech2 model.
"""

import paddle.v2 as paddle
from itertools import groupby
import distutils.util
import argparse
import gzip
from audio_data_utils import DataGenerator
from model import deep_speech2

parser = argparse.ArgumentParser(
    description='Simplified version of DeepSpeech2 inference.')
parser.add_argument(
    "--num_samples",
    default=10,
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
    "--normalizer_manifest_path",
    default='./manifest.libri.train-clean-100',
    type=str,
    help="Manifest path for normalizer. (default: %(default)s)")
parser.add_argument(
    "--decode_manifest_path",
    default='./manifest.libri.test-clean',
    type=str,
    help="Manifest path for decoding. (default: %(default)s)")
parser.add_argument(
    "--model_filepath",
    default='./params.tar.gz',
    type=str,
    help="Model filepath. (default: %(default)s)")
args = parser.parse_args()


def remove_duplicate_and_blank(id_list, blank_id):
    """
    Postprocessing for max-ctc-decoder.
    - remove consecutive duplicate tokens.
    - remove blanks.
    """
    # remove consecutive duplicate tokens
    id_list = [x[0] for x in groupby(id_list)]
    # remove blanks
    return [id for id in id_list if id != blank_id]


def best_path_decode():
    """
    Max-ctc-decoding for DeepSpeech2.
    """
    # initialize data generator
    data_generator = DataGenerator(
        vocab_filepath='eng_vocab.txt',
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
    _, max_id = deep_speech2(
        audio_data=audio_data,
        text_data=text_data,
        dict_size=dict_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_size=args.rnn_layer_size)

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

    # run max-ctc-decoding
    max_id_results = paddle.infer(
        output_layer=max_id,
        parameters=parameters,
        input=infer_data,
        field=['id'])

    # postprocess
    instance_length = len(max_id_results) / args.num_samples
    instance_list = [
        max_id_results[i * instance_length:(i + 1) * instance_length]
        for i in xrange(0, args.num_samples)
    ]
    for i, instance in enumerate(instance_list):
        id_list = remove_duplicate_and_blank(instance, dict_size)
        output_transcript = ''.join([vocab_list[id] for id in id_list])
        target_transcript = ''.join([vocab_list[id] for id in infer_data[i][1]])
        print("Target Transcript: %s \nOutput Transcript: %s \n" %
              (target_transcript, output_transcript))


def main():
    paddle.init(use_gpu=args.use_gpu, trainer_count=1)
    best_path_decode()


if __name__ == '__main__':
    main()
