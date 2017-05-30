"""
   Trainer for a simplifed version of Baidu DeepSpeech2 model.
"""

import paddle.v2 as paddle
import argparse
import gzip
import time
import sys
from model import deep_speech2
from audio_data_utils import DataGenerator
import numpy as np

#TODO: add WER metric

parser = argparse.ArgumentParser(
    description='Simplified version of DeepSpeech2 trainer.')
parser.add_argument(
    "--batch_size", default=32, type=int, help="Minibatch size.")
parser.add_argument("--trainer", default=1, type=int, help="Trainer number.")
parser.add_argument(
    "--num_passes", default=20, type=int, help="Training pass number.")
parser.add_argument(
    "--num_conv_layers", default=3, type=int, help="Convolution layer number.")
parser.add_argument(
    "--num_rnn_layers", default=5, type=int, help="RNN layer number.")
parser.add_argument(
    "--rnn_layer_size", default=512, type=int, help="RNN layer cell number.")
parser.add_argument(
    "--use_gpu", default=True, type=bool, help="Use gpu or not.")
parser.add_argument(
    "--use_sortagrad", default=False, type=bool, help="Use sortagrad or not.")
parser.add_argument(
    "--trainer_count", default=8, type=int, help="Trainer number.")
args = parser.parse_args()


def train():
    """
    DeepSpeech2 training.
    """
    # create data readers
    data_generator = DataGenerator(
        vocab_filepath='eng_vocab.txt',
        normalizer_manifest_path='./libri.manifest.train',
        normalizer_num_samples=200,
        max_duration=20.0,
        min_duration=0.0,
        stride_ms=10,
        window_ms=20)
    train_batch_reader_sortagrad = data_generator.batch_reader_creator(
        manifest_path='./libri.manifest.dev.small',
        batch_size=args.batch_size // args.trainer,
        padding_to=2000,
        flatten=True,
        sort_by_duration=True,
        shuffle=False)
    train_batch_reader_nosortagrad = data_generator.batch_reader_creator(
        manifest_path='./libri.manifest.dev.small',
        batch_size=args.batch_size // args.trainer,
        padding_to=2000,
        flatten=True,
        sort_by_duration=False,
        shuffle=True)
    test_batch_reader = data_generator.batch_reader_creator(
        manifest_path='./libri.manifest.test',
        batch_size=args.batch_size // args.trainer,
        padding_to=2000,
        flatten=True,
        sort_by_duration=False,
        shuffle=False)
    feeding = data_generator.data_name_feeding()

    # create network config
    dict_size = data_generator.vocabulary_size()
    audio_data = paddle.layer.data(
        name="audio_spectrogram",
        height=161,
        width=2000,
        type=paddle.data_type.dense_vector(322000))
    text_data = paddle.layer.data(
        name="transcript_text",
        type=paddle.data_type.integer_value_sequence(dict_size))
    cost, _ = deep_speech2(
        audio_data=audio_data,
        text_data=text_data,
        dict_size=dict_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_size=args.rnn_layer_size)

    # create parameters and optimizer
    parameters = paddle.parameters.create(cost)
    optimizer = paddle.optimizer.Adam(
        learning_rate=5e-5, gradient_clipping_threshold=400)
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # create event handler
    def event_handler(event):
        global start_time
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 10 == 0:
                print "\nPass: %d, Batch: %d, TrainCost: %f" % (
                    event.pass_id, event.batch_id, event.cost)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.BeginPass):
            start_time = time.time()
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_batch_reader, feeding=feeding)
            print "\n------- Time: %d,  Pass: %d, TestCost: %s" % (
                time.time() - start_time, event.pass_id, result.cost)
            with gzip.open("params.tar.gz", 'w') as f:
                parameters.to_tar(f)

    # run train
    # first pass with sortagrad
    if args.use_sortagrad:
        trainer.train(
            reader=train_batch_reader_sortagrad,
            event_handler=event_handler,
            num_passes=1,
            feeding=feeding)
        args.num_passes -= 1
    # other passes without sortagrad
    trainer.train(
        reader=train_batch_reader_nosortagrad,
        event_handler=event_handler,
        num_passes=args.num_passes,
        feeding=feeding)


def main():
    paddle.init(use_gpu=args.use_gpu, trainer_count=args.trainer_count)
    train()


if __name__ == '__main__':
    main()
