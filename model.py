"""Contains DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import gzip
from decoder import *
from lm.lm_scorer import LmScorer
import paddle.v2 as paddle
from layer import *


class DeepSpeech2Model(object):
    """DeepSpeech2Model class.

    :param vocab_size: Decoding vocabulary size.
    :type vocab_size: int
    :param num_conv_layers: Number of stacking convolution layers.
    :type num_conv_layers: int
    :param num_rnn_layers: Number of stacking RNN layers.
    :type num_rnn_layers: int
    :param rnn_layer_size: RNN layer size (number of RNN cells).
    :type rnn_layer_size: int
    :param pretrained_model_path: Pretrained model path. If None, will train
                                  from stratch.
    :type pretrained_model_path: basestring|None
    """

    def __init__(self, vocab_size, num_conv_layers, num_rnn_layers,
                 rnn_layer_size, pretrained_model_path):
        self._create_network(vocab_size, num_conv_layers, num_rnn_layers,
                             rnn_layer_size)
        self._create_parameters(pretrained_model_path)
        self._inferer = None
        self._loss_inferer = None
        self._ext_scorer = None

    def train(self,
              train_batch_reader,
              dev_batch_reader,
              feeding_dict,
              learning_rate,
              gradient_clipping,
              num_passes,
              output_model_dir,
              num_iterations_print=100):
        """Train the model.

        :param train_batch_reader: Train data reader.
        :type train_batch_reader: callable
        :param dev_batch_reader: Validation data reader.
        :type dev_batch_reader: callable
        :param feeding_dict: Feeding is a map of field name and tuple index
                             of the data that reader returns.
        :type feeding_dict: dict|list
        :param learning_rate: Learning rate for ADAM optimizer.
        :type learning_rate: float
        :param gradient_clipping: Gradient clipping threshold.
        :type gradient_clipping: float
        :param num_passes: Number of training epochs.
        :type num_passes: int
        :param num_iterations_print: Number of training iterations for printing
                                     a training loss.
        :type rnn_iteratons_print: int
        :param output_model_dir: Directory for saving the model (every pass).
        :type output_model_dir: basestring
        """
        # prepare model output directory
        if not os.path.exists(output_model_dir):
            os.mkdir(output_model_dir)

        # prepare optimizer and trainer
        optimizer = paddle.optimizer.Adam(
            learning_rate=learning_rate,
            gradient_clipping_threshold=gradient_clipping)
        trainer = paddle.trainer.SGD(
            cost=self._loss,
            parameters=self._parameters,
            update_equation=optimizer)

        # create event handler
        def event_handler(event):
            global start_time, cost_sum, cost_counter
            if isinstance(event, paddle.event.EndIteration):
                cost_sum += event.cost
                cost_counter += 1
                if (event.batch_id + 1) % num_iterations_print == 0:
                    output_model_path = os.path.join(output_model_dir,
                                                     "params.latest.tar.gz")
                    with gzip.open(output_model_path, 'w') as f:
                        self._parameters.to_tar(f)
                    print("\nPass: %d, Batch: %d, TrainCost: %f" %
                          (event.pass_id, event.batch_id + 1,
                           cost_sum / cost_counter))
                    cost_sum, cost_counter = 0.0, 0
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
            if isinstance(event, paddle.event.BeginPass):
                start_time = time.time()
                cost_sum, cost_counter = 0.0, 0
            if isinstance(event, paddle.event.EndPass):
                result = trainer.test(
                    reader=dev_batch_reader, feeding=feeding_dict)
                output_model_path = os.path.join(
                    output_model_dir, "params.pass-%d.tar.gz" % event.pass_id)
                with gzip.open(output_model_path, 'w') as f:
                    self._parameters.to_tar(f)
                print("\n------- Time: %d sec,  Pass: %d, ValidationCost: %s" %
                      (time.time() - start_time, event.pass_id, result.cost))

        # run train
        trainer.train(
            reader=train_batch_reader,
            event_handler=event_handler,
            num_passes=num_passes,
            feeding=feeding_dict)

    def infer_loss_batch(self, infer_data):
        # define inferer
        if self._loss_inferer == None:
            self._loss_inferer = paddle.inference.Inference(
                output_layer=self._loss, parameters=self._parameters)
        # run inference
        return self._loss_inferer.infer(input=infer_data)

    def infer_batch(self, infer_data, decode_method, beam_alpha, beam_beta,
                    beam_size, cutoff_prob, vocab_list, language_model_path,
                    num_processes):
        """Model inference. Infer the transcription for a batch of speech
        utterances.

        :param infer_data: List of utterances to infer, with each utterance a
                           tuple of audio features and transcription text (empty
                           string).
        :type infer_data: list
        :param decode_method: Decoding method name, 'best_path' or
                              'beam search'.
        :param decode_method: string
        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param beam_size: Width for Beam search.
        :type beam_size: int
        :param cutoff_prob: Cutoff probability in pruning,
                            default 1.0, no pruning.
        :type cutoff_prob: float
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        :param language_model_path: Filepath for language model.
        :type language_model_path: basestring|None
        :param num_processes: Number of processes (CPU) for decoder.
        :type num_processes: int
        :return: List of transcription texts.
        :rtype: List of basestring
        """
        # define inferer
        if self._inferer == None:
            self._inferer = paddle.inference.Inference(
                output_layer=self._log_probs, parameters=self._parameters)
        # run inference
        infer_results = self._inferer.infer(input=infer_data)
        num_steps = len(infer_results) // len(infer_data)
        probs_split = [
            infer_results[i * num_steps:(i + 1) * num_steps]
            for i in xrange(0, len(infer_data))
        ]
        # run decoder
        results = []
        if decode_method == "best_path":
            # best path decode
            for i, probs in enumerate(probs_split):
                output_transcription = ctc_best_path_decoder(
                    probs_seq=probs, vocabulary=data_generator.vocab_list)
                results.append(output_transcription)
        elif decode_method == "beam_search":
            # initialize external scorer
            if self._ext_scorer == None:
                self._ext_scorer = LmScorer(beam_alpha, beam_beta,
                                            language_model_path)
                self._loaded_lm_path = language_model_path
            else:
                self._ext_scorer.reset_params(beam_alpha, beam_beta)
                assert self._loaded_lm_path == language_model_path

            # beam search decode
            beam_search_results = ctc_beam_search_decoder_batch(
                probs_split=probs_split,
                vocabulary=vocab_list,
                beam_size=beam_size,
                blank_id=len(vocab_list),
                num_processes=num_processes,
                ext_scoring_func=self._ext_scorer,
                cutoff_prob=cutoff_prob)

            results = [result[0][1] for result in beam_search_results]
        else:
            raise ValueError("Decoding method [%s] is not supported." %
                             decode_method)
        return results

    def _create_parameters(self, model_path=None):
        """Load or create model parameters."""
        if model_path is None:
            self._parameters = paddle.parameters.create(self._loss)
        else:
            self._parameters = paddle.parameters.Parameters.from_tar(
                gzip.open(model_path))

    def _create_network(self, vocab_size, num_conv_layers, num_rnn_layers,
                        rnn_layer_size):
        """Create data layers and model network."""
        # paddle.data_type.dense_array is used for variable batch input.
        # The size 161 * 161 is only an placeholder value and the real shape
        # of input batch data will be induced during training.
        audio_data = paddle.layer.data(
            name="audio_spectrogram",
            type=paddle.data_type.dense_array(161 * 161))
        text_data = paddle.layer.data(
            name="transcript_text",
            type=paddle.data_type.integer_value_sequence(vocab_size))
        self._log_probs, self._loss = deep_speech2(
            audio_data=audio_data,
            text_data=text_data,
            dict_size=vocab_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            rnn_size=rnn_layer_size)
