"""Contains DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import logging
import gzip
import copy
import inspect
from distutils.dir_util import mkpath
import paddle.v2 as paddle
from decoders.swig_wrapper import Scorer
from decoders.swig_wrapper import ctc_greedy_decoder
from decoders.swig_wrapper import ctc_beam_search_decoder_batch
from model_utils.network import deep_speech_v2_network

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')


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
    :param share_rnn_weights: Whether to share input-hidden weights between
                              forward and backward directional RNNs.Notice that
                              for GRU, weight sharing is not supported.
    :type share_rnn_weights: bool
    """

    def __init__(self, vocab_size, num_conv_layers, num_rnn_layers,
                 rnn_layer_size, use_gru, pretrained_model_path,
                 share_rnn_weights):
        self._create_network(vocab_size, num_conv_layers, num_rnn_layers,
                             rnn_layer_size, use_gru, share_rnn_weights)
        self._create_parameters(pretrained_model_path)
        self._inferer = None
        self._loss_inferer = None
        self._ext_scorer = None
        self._num_conv_layers = num_conv_layers
        self.logger = logging.getLogger("")
        self.logger.setLevel(level=logging.INFO)

    def train(self,
              train_batch_reader,
              dev_batch_reader,
              feeding_dict,
              learning_rate,
              gradient_clipping,
              num_passes,
              output_model_dir,
              is_local=True,
              num_iterations_print=100,
              test_off=False):
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
        :param is_local: Set to False if running with pserver with multi-nodes.
        :type is_local: bool
        :param output_model_dir: Directory for saving the model (every pass).
        :type output_model_dir: basestring
        :param test_off: Turn off testing.
        :type test_off: bool
        """
        # prepare model output directory
        if not os.path.exists(output_model_dir):
            mkpath(output_model_dir)

        # adapt the feeding dict and reader according to the network
        adapted_feeding_dict = self._adapt_feeding_dict(feeding_dict)
        adapted_train_batch_reader = self._adapt_data(train_batch_reader)
        adapted_dev_batch_reader = self._adapt_data(dev_batch_reader)

        # prepare optimizer and trainer
        optimizer = paddle.optimizer.Adam(
            learning_rate=learning_rate,
            gradient_clipping_threshold=gradient_clipping)
        trainer = paddle.trainer.SGD(
            cost=self._loss,
            parameters=self._parameters,
            update_equation=optimizer,
            is_local=is_local)

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
                        trainer.save_parameter_to_tar(f)
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
                if test_off:
                    print("\n------- Time: %d sec,  Pass: %d" %
                          (time.time() - start_time, event.pass_id))
                else:
                    result = trainer.test(
                        reader=adapted_dev_batch_reader,
                        feeding=adapted_feeding_dict)
                    print(
                        "\n------- Time: %d sec,  Pass: %d, "
                        "ValidationCost: %s" %
                        (time.time() - start_time, event.pass_id, result.cost))
                output_model_path = os.path.join(
                    output_model_dir, "params.pass-%d.tar.gz" % event.pass_id)
                with gzip.open(output_model_path, 'w') as f:
                    trainer.save_parameter_to_tar(f)

        # run train
        trainer.train(
            reader=adapted_train_batch_reader,
            event_handler=event_handler,
            num_passes=num_passes,
            feeding=adapted_feeding_dict)

    # TODO(@pkuyym) merge this function into infer_batch
    def infer_loss_batch(self, infer_data):
        """Model inference. Infer the ctc loss for a batch of speech
        utterances.

        :param infer_data: List of utterances to infer, with each utterance a
                           tuple of audio features and transcription text (empty
                           string).
        :type infer_data: list
        :return: List of ctc loss.
        :rtype: List of float
        """
        # define inferer
        if self._loss_inferer == None:
            self._loss_inferer = paddle.inference.Inference(
                output_layer=self._loss, parameters=self._parameters)
        # run inference
        return self._loss_inferer.infer(input=infer_data)

    def infer_batch_probs(self, infer_data, feeding_dict):
        """Infer the prob matrices for a batch of speech utterances.

        :param infer_data: List of utterances to infer, with each utterance
                           consisting of a tuple of audio features and
                           transcription text (empty string).
        :type infer_data: list
        :param feeding_dict: Feeding is a map of field name and tuple index
                             of the data that reader returns.
        :type feeding_dict: dict|list
        :return: List of 2-D probability matrix, and each consists of prob
                 vectors for one speech utterancce.
        :rtype: List of matrix
        """
        # define inferer
        if self._inferer == None:
            self._inferer = paddle.inference.Inference(
                output_layer=self._log_probs, parameters=self._parameters)
        adapted_feeding_dict = self._adapt_feeding_dict(feeding_dict)
        adapted_infer_data = self._adapt_data(infer_data)
        # run inference
        infer_results = self._inferer.infer(
            input=adapted_infer_data, feeding=adapted_feeding_dict)
        start_pos = [0] * (len(adapted_infer_data) + 1)
        for i in xrange(len(adapted_infer_data)):
            start_pos[i + 1] = start_pos[i] + adapted_infer_data[i][3][0]
        probs_split = [
            infer_results[start_pos[i]:start_pos[i + 1]]
            for i in xrange(0, len(adapted_infer_data))
        ]
        return probs_split

    def decode_batch_greedy(self, probs_split, vocab_list):
        """Decode by best path for a batch of probs matrix input.

        :param probs_split: List of 2-D probability matrix, and each consists
                            of prob vectors for one speech utterancce.
        :param probs_split: List of matrix
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        :return: List of transcription texts.
        :rtype: List of basestring
        """
        results = []
        for i, probs in enumerate(probs_split):
            output_transcription = ctc_greedy_decoder(
                probs_seq=probs, vocabulary=vocab_list)
            results.append(output_transcription)
        return results

    def init_ext_scorer(self, beam_alpha, beam_beta, language_model_path,
                        vocab_list):
        """Initialize the external scorer.

        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param language_model_path: Filepath for language model. If it is
                                    empty, the external scorer will be set to
                                    None, and the decoding method will be pure
                                    beam search without scorer.
        :type language_model_path: basestring|None
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        """
        if language_model_path != '':
            self.logger.info("begin to initialize the external scorer "
                             "for decoding")
            self._ext_scorer = Scorer(beam_alpha, beam_beta,
                                      language_model_path, vocab_list)
            lm_char_based = self._ext_scorer.is_character_based()
            lm_max_order = self._ext_scorer.get_max_order()
            lm_dict_size = self._ext_scorer.get_dict_size()
            self.logger.info("language model: "
                             "is_character_based = %d," % lm_char_based +
                             " max_order = %d," % lm_max_order +
                             " dict_size = %d" % lm_dict_size)
            self.logger.info("end initializing scorer")
        else:
            self._ext_scorer = None
            self.logger.info("no language model provided, "
                             "decoding by pure beam search without scorer.")

    def decode_batch_beam_search(self, probs_split, beam_alpha, beam_beta,
                                 beam_size, cutoff_prob, cutoff_top_n,
                                 vocab_list, num_processes):
        """Decode by beam search for a batch of probs matrix input.

        :param probs_split: List of 2-D probability matrix, and each consists
                            of prob vectors for one speech utterancce.
        :param probs_split: List of matrix
        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param beam_size: Width for Beam search.
        :type beam_size: int
        :param cutoff_prob: Cutoff probability in pruning,
                            default 1.0, no pruning.
        :type cutoff_prob: float
        :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                        characters with highest probs in vocabulary will be
                        used in beam search, default 40.
        :type cutoff_top_n: int
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        :param num_processes: Number of processes (CPU) for decoder.
        :type num_processes: int
        :return: List of transcription texts.
        :rtype: List of basestring
        """
        if self._ext_scorer != None:
            self._ext_scorer.reset_params(beam_alpha, beam_beta)
        # beam search decode
        num_processes = min(num_processes, len(probs_split))
        beam_search_results = ctc_beam_search_decoder_batch(
            probs_split=probs_split,
            vocabulary=vocab_list,
            beam_size=beam_size,
            num_processes=num_processes,
            ext_scoring_func=self._ext_scorer,
            cutoff_prob=cutoff_prob,
            cutoff_top_n=cutoff_top_n)

        results = [result[0][1] for result in beam_search_results]
        return results

    def _adapt_feeding_dict(self, feeding_dict):
        """Adapt feeding dict according to network struct.

        To remove impacts from padding part, we add scale_sub_region layer and
        sub_seq layer. For sub_seq layer, 'sequence_offset' and
        'sequence_length' fields are appended. For each scale_sub_region layer
        'convN_index_range' field is appended.

        :param feeding_dict: Feeding is a map of field name and tuple index
                             of the data that reader returns.
        :type feeding_dict: dict|list
        :return: Adapted feeding dict.
        :rtype: dict|list
        """
        adapted_feeding_dict = copy.deepcopy(feeding_dict)
        if isinstance(feeding_dict, dict):
            adapted_feeding_dict["sequence_offset"] = len(adapted_feeding_dict)
            adapted_feeding_dict["sequence_length"] = len(adapted_feeding_dict)
            for i in xrange(self._num_conv_layers):
                adapted_feeding_dict["conv%d_index_range" %i] = \
                        len(adapted_feeding_dict)
        elif isinstance(feeding_dict, list):
            adapted_feeding_dict.append("sequence_offset")
            adapted_feeding_dict.append("sequence_length")
            for i in xrange(self._num_conv_layers):
                adapted_feeding_dict.append("conv%d_index_range" % i)
        else:
            raise ValueError("Type of feeding_dict is %s, not supported." %
                             type(feeding_dict))

        return adapted_feeding_dict

    def _adapt_data(self, data):
        """Adapt data according to network struct.

        For each convolution layer in the conv_group, to remove impacts from
        padding data, we can multiply zero to the padding part of the outputs
        of each batch normalization layer. We add a scale_sub_region layer after
        each batch normalization layer to reset the padding data.
        For rnn layers, to remove impacts from padding data, we can truncate the
        padding part before output data feeded into the first rnn layer. We use
        sub_seq layer to achieve this.

        :param data: Data from data_provider.
        :type data: list|function
        :return: Adapted data.
        :rtype: list|function
        """

        def adapt_instance(instance):
            if len(instance) < 2 or len(instance) > 3:
                raise ValueError("Size of instance should be 2 or 3.")
            padded_audio = instance[0]
            text = instance[1]
            # no padding part
            if len(instance) == 2:
                audio_len = padded_audio.shape[1]
            else:
                audio_len = instance[2]
            adapted_instance = [padded_audio, text]
            # Stride size for conv0 is (3, 2)
            # Stride size for conv1 to convN is (1, 2)
            # Same as the network, hard-coded here
            padded_conv0_h = (padded_audio.shape[0] - 1) // 2 + 1
            padded_conv0_w = (padded_audio.shape[1] - 1) // 3 + 1
            valid_w = (audio_len - 1) // 3 + 1
            adapted_instance += [
                [0],  # sequence offset, always 0
                [valid_w],  # valid sequence length
                # Index ranges for channel, height and width
                # Please refer scale_sub_region layer to see details
                [1, 32, 1, padded_conv0_h, valid_w + 1, padded_conv0_w]
            ]
            pre_padded_h = padded_conv0_h
            for i in xrange(self._num_conv_layers - 1):
                padded_h = (pre_padded_h - 1) // 2 + 1
                pre_padded_h = padded_h
                adapted_instance += [
                    [1, 32, 1, padded_h, valid_w + 1, padded_conv0_w]
                ]
            return adapted_instance

        if isinstance(data, list):
            return map(adapt_instance, data)
        elif inspect.isgeneratorfunction(data):

            def adapted_reader():
                for instance in data():
                    yield map(adapt_instance, instance)

            return adapted_reader
        else:
            raise ValueError("Type of data is %s, not supported." % type(data))

    def _create_parameters(self, model_path=None):
        """Load or create model parameters."""
        if model_path is None:
            self._parameters = paddle.parameters.create(self._loss)
        else:
            self._parameters = paddle.parameters.Parameters.from_tar(
                gzip.open(model_path))

    def _create_network(self, vocab_size, num_conv_layers, num_rnn_layers,
                        rnn_layer_size, use_gru, share_rnn_weights):
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
        seq_offset_data = paddle.layer.data(
            name='sequence_offset',
            type=paddle.data_type.integer_value_sequence(1))
        seq_len_data = paddle.layer.data(
            name='sequence_length',
            type=paddle.data_type.integer_value_sequence(1))
        index_range_datas = []
        for i in xrange(num_rnn_layers):
            index_range_datas.append(
                paddle.layer.data(
                    name='conv%d_index_range' % i,
                    type=paddle.data_type.dense_vector(6)))

        self._log_probs, self._loss = deep_speech_v2_network(
            audio_data=audio_data,
            text_data=text_data,
            seq_offset_data=seq_offset_data,
            seq_len_data=seq_len_data,
            index_range_datas=index_range_datas,
            dict_size=vocab_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            rnn_size=rnn_layer_size,
            use_gru=use_gru,
            share_rnn_weights=share_rnn_weights)
