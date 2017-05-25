"""
   Inference for a simplifed version of Baidu DeepSpeech2 model.
"""

import paddle.v2 as paddle
from itertools import groupby
import argparse
import gzip
import audio_data_utils
from model import deep_speech2

parser = argparse.ArgumentParser(
    description='Simplified version of DeepSpeech2 inference.')
parser.add_argument(
    "--num_samples",
    default=10,
    type=int,
    help="Number of samples for inference.")
parser.add_argument(
    "--num_conv_layers", default=2, type=int, help="Convolution layer number.")
parser.add_argument(
    "--num_rnn_layers", default=3, type=int, help="RNN layer number.")
parser.add_argument(
    "--rnn_layer_size", default=512, type=int, help="RNN layer cell number.")
parser.add_argument(
    "--use_gpu", default=True, type=bool, help="Use gpu or not.")
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


def max_infer():
    """
    Max-ctc-decoding for DeepSpeech2.
    """
    # create network config
    _, vocab_list = audio_data_utils.get_vocabulary()
    dict_size = len(vocab_list)
    audio_data = paddle.layer.data(
        name="audio_spectrogram",
        height=161,
        width=1000,
        type=paddle.data_type.dense_vector(161000))
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
        gzip.open("params.tar.gz"))

    # prepare infer data
    feeding = {
        "audio_spectrogram": 0,
        "transcript_text": 1,
    }
    test_batch_reader = audio_data_utils.padding_batch_reader(
        paddle.batch(
            audio_data_utils.reader_creator(
                manifest_path="./libri.manifest.test", sort_by_duration=False),
            batch_size=args.num_samples),
        padding=[-1, 1000])
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
        max_id_results[i:i + instance_length]
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
    max_infer()


if __name__ == '__main__':
    main()
