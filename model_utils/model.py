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
import cPickle as pickle
import collections
import multiprocessing
import numpy as np
from distutils.dir_util import mkpath
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
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
    :param use_gru: Use gru if set True. Use simple rnn if set False.
    :type use_gru: bool
    :param share_rnn_weights: Whether to share input-hidden weights between
                              forward and backward directional RNNs.Notice that
                              for GRU, weight sharing is not supported.
    :type share_rnn_weights: bool
    :param place: Program running place.
    :type place: CPUPlace or CUDAPlace
    :param init_from_pretrained_model: Pretrained model path. If None, will train
                                  from stratch.
    :type init_from_pretrained_model: string|None
    :param output_model_dir: Output model directory. If None, output to current directory. 
    :type output_model_dir: string|None
    """

    def __init__(self,
                 vocab_size,
                 num_conv_layers,
                 num_rnn_layers,
                 rnn_layer_size,
                 use_gru=False,
                 share_rnn_weights=True,
                 place=fluid.CPUPlace(),
                 init_from_pretrained_model=None,
                 output_model_dir=None):
        self._vocab_size = vocab_size
        self._num_conv_layers = num_conv_layers
        self._num_rnn_layers = num_rnn_layers
        self._rnn_layer_size = rnn_layer_size
        self._use_gru = use_gru
        self._share_rnn_weights = share_rnn_weights
        self._place = place
        self._init_from_pretrained_model = init_from_pretrained_model
        self._output_model_dir = output_model_dir
        self._ext_scorer = None
        self.logger = logging.getLogger("")
        self.logger.setLevel(level=logging.INFO)

    def create_network(self, is_infer=False):
        """Create data layers and model network.
        :param is_training: Whether to create a network for training.
        :type is_training: bool 
        :return reader: Reader for input.
        :rtype reader: read generater
        :return log_probs: An output unnormalized log probability layer.
        :rtype lig_probs: Varable
        :return loss: A ctc loss layer.
        :rtype loss: Variable
        """

        if not is_infer:
            input_fields = {
                'names': ['audio_data', 'text_data', 'seq_len_data', 'masks'],
                'shapes':
                [[None, 161, None], [None, 1], [None, 1], [None, 32, 81, None]],
                'dtypes': ['float32', 'int32', 'int64', 'float32'],
                'lod_levels': [0, 1, 0, 0]
            }

            inputs = [
                fluid.data(
                    name=input_fields['names'][i],
                    shape=input_fields['shapes'][i],
                    dtype=input_fields['dtypes'][i],
                    lod_level=input_fields['lod_levels'][i])
                for i in range(len(input_fields['names']))
            ]

            reader = fluid.io.DataLoader.from_generator(
                feed_list=inputs,
                capacity=64,
                iterable=False,
                use_double_buffer=True)

            (audio_data, text_data, seq_len_data, masks) = inputs
        else:
            audio_data = fluid.data(
                name='audio_data',
                shape=[None, 161, None],
                dtype='float32',
                lod_level=0)
            seq_len_data = fluid.data(
                name='seq_len_data',
                shape=[None, 1],
                dtype='int64',
                lod_level=0)
            masks = fluid.data(
                name='masks',
                shape=[None, 32, 81, None],
                dtype='float32',
                lod_level=0)
            text_data = None
            reader = fluid.DataFeeder([audio_data, seq_len_data, masks],
                                      self._place)

        log_probs, loss = deep_speech_v2_network(
            audio_data=audio_data,
            text_data=text_data,
            seq_len_data=seq_len_data,
            masks=masks,
            dict_size=self._vocab_size,
            num_conv_layers=self._num_conv_layers,
            num_rnn_layers=self._num_rnn_layers,
            rnn_size=self._rnn_layer_size,
            use_gru=self._use_gru,
            share_rnn_weights=self._share_rnn_weights)
        return reader, log_probs, loss

    def init_from_pretrained_model(self, exe, program):
        '''Init params from pretrain model. '''

        assert isinstance(self._init_from_pretrained_model, str)

        if not os.path.exists(self._init_from_pretrained_model):
            print(self._init_from_pretrained_model)
            raise Warning("The pretrained params do not exist.")
            return False
        fluid.io.load_params(
            exe,
            self._init_from_pretrained_model,
            main_program=program,
            filename="params.pdparams")

        print("finish initing model from pretrained params from %s" %
              (self._init_from_pretrained_model))

        pre_epoch = 0
        dir_name = self._init_from_pretrained_model.split('_')
        if len(dir_name) >= 2 and dir_name[-2].endswith('epoch') and dir_name[
                -1].isdigit():
            pre_epoch = int(dir_name[-1])

        return pre_epoch + 1

    def save_param(self, exe, program, dirname):
        '''Save model params to dirname'''

        assert isinstance(self._output_model_dir, str)

        param_dir = os.path.join(self._output_model_dir)

        if not os.path.exists(param_dir):
            os.mkdir(param_dir)

        fluid.io.save_params(
            exe,
            os.path.join(param_dir, dirname),
            main_program=program,
            filename="params.pdparams")
        print("save parameters at %s" % (os.path.join(param_dir, dirname)))

        return True

    def test(self, exe, dev_batch_reader, test_program, test_reader,
             fetch_list):
        '''Test the model.

        :param exe:The executor of program.
        :type exe: Executor
        :param dev_batch_reader: The reader of test dataa.
        :type dev_batch_reader: read generator 
        :param test_program: The program of test.
        :type test_program: Program
        :param test_reader: Reader of test.
        :type test_reader: Reader
        :param fetch_list: Fetch list.
        :type fetch_list: list
        :return: An output unnormalized log probability. 
        :rtype: array
        '''
        test_reader.start()
        epoch_loss = []
        while True:
            try:
                each_loss = exe.run(
                    program=test_program,
                    fetch_list=fetch_list,
                    return_numpy=False)
                epoch_loss.extend(np.array(each_loss[0]))

            except fluid.core.EOFException:
                test_reader.reset()
                break
        return np.mean(np.array(epoch_loss))

    def train(self,
              train_batch_reader,
              dev_batch_reader,
              feeding_dict,
              learning_rate,
              gradient_clipping,
              num_epoch,
              batch_size,
              num_samples,
              save_epoch=100,
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
        :param num_epoch: Number of training epochs.
        :type num_epoch: int
        :param batch_size: Number of batch size.
        :type batch_size: int
        :param num_samples: The num of train samples.
        :type num_samples: int
        :param save_epoch: Number of training iterations for save checkpoint and params.
        :type save_epoch: int
        :param num_iterations_print: Number of training iterations for printing
                                     a training loss.
        :type num_iteratons_print: int
        :param test_off: Turn off testing.
        :type test_off: bool
        """
        # prepare model output directory
        if not os.path.exists(self._output_model_dir):
            mkpath(self._output_model_dir)

        # adapt the feeding dict according to the network
        adapted_feeding_dict = self._adapt_feeding_dict(feeding_dict)

        if isinstance(self._place, fluid.CUDAPlace):
            dev_count = fluid.core.get_cuda_device_count()
        else:
            dev_count = int(os.environ.get('CPU_NUM', 1))

        # prepare the network
        train_program = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_reader, log_probs, ctc_loss = self.create_network()
                # prepare optimizer
                optimizer = fluid.optimizer.AdamOptimizer(
                    learning_rate=fluid.layers.exponential_decay(
                        learning_rate=learning_rate,
                        decay_steps=num_samples / batch_size / dev_count,
                        decay_rate=0.83,
                        staircase=True))
                fluid.clip.set_gradient_clip(
                    clip=fluid.clip.GradientClipByGlobalNorm(
                        clip_norm=gradient_clipping))
                optimizer.minimize(loss=ctc_loss)

        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_reader, _, ctc_loss = self.create_network()

        test_prog = test_prog.clone(for_test=True)

        exe = fluid.Executor(self._place)
        exe.run(startup_prog)

        # init from some pretrain models, to better solve the current task
        pre_epoch = 0
        if self._init_from_pretrained_model:
            pre_epoch = self.init_from_pretrained_model(exe, train_program)

        build_strategy = compiler.BuildStrategy()
        exec_strategy = fluid.ExecutionStrategy()

        # pass the build_strategy to with_data_parallel API
        compiled_prog = compiler.CompiledProgram(
            train_program).with_data_parallel(
                loss_name=ctc_loss.name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)

        train_reader.set_batch_generator(train_batch_reader)
        test_reader.set_batch_generator(dev_batch_reader)

        # run train 
        for epoch_id in range(num_epoch):
            train_reader.start()
            epoch_loss = []
            time_begin = time.time()
            batch_id = 0
            step = 0
            while True:
                try:
                    fetch_list = [ctc_loss.name]

                    if batch_id % num_iterations_print == 0:
                        fetch = exe.run(
                            program=compiled_prog,
                            fetch_list=fetch_list,
                            return_numpy=False)
                        each_loss = fetch[0]
                        epoch_loss.extend(np.array(each_loss[0]) / batch_size)

                        print("epoch: %d, batch: %d, train loss: %f\n" %
                              (epoch_id, batch_id,
                               np.mean(each_loss[0]) / batch_size))

                    else:
                        each_loss = exe.run(
                            program=compiled_prog,
                            fetch_list=[],
                            return_numpy=False)

                    batch_id = batch_id + 1
                except fluid.core.EOFException:
                    train_reader.reset()
                    break
            time_end = time.time()
            used_time = time_end - time_begin
            if test_off:
                print("\n--------Time: %f sec, epoch: %d, train loss: %f\n" %
                      (used_time, epoch_id, np.mean(np.array(epoch_loss))))
            else:
                print('\n----------Begin test...')
                test_loss = self.test(
                    exe,
                    dev_batch_reader=dev_batch_reader,
                    test_program=test_prog,
                    test_reader=test_reader,
                    fetch_list=[ctc_loss])
                print(
                    "--------Time: %f sec, epoch: %d, train loss: %f, test loss: %f"
                    % (used_time, epoch_id + pre_epoch,
                       np.mean(np.array(epoch_loss)), test_loss / batch_size))
            if (epoch_id + 1) % save_epoch == 0:
                self.save_param(exe, train_program,
                                "epoch_" + str(epoch_id + pre_epoch))

        self.save_param(exe, train_program, "step_final")

        print("\n------------Training finished!!!-------------")

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
        infer_program = fluid.Program()
        startup_prog = fluid.Program()

        # adapt the feeding dict according to the network
        adapted_feeding_dict = self._adapt_feeding_dict(feeding_dict)

        # prepare the network
        with fluid.program_guard(infer_program, startup_prog):
            with fluid.unique_name.guard():
                feeder, log_probs, _ = self.create_network(is_infer=True)

        infer_program = infer_program.clone(for_test=True)
        exe = fluid.Executor(self._place)
        exe.run(startup_prog)

        # init param from pretrained_model
        if not self._init_from_pretrained_model:
            exit("No pretrain model file path!")
        self.init_from_pretrained_model(exe, infer_program)

        infer_results = []
        time_begin = time.time()

        # run inference
        for i in range(infer_data[0].shape[0]):
            each_log_probs = exe.run(
                program=infer_program,
                feed=feeder.feed(
                    [[infer_data[0][i], infer_data[2][i], infer_data[3][i]]]),
                fetch_list=[log_probs],
                return_numpy=False)
            infer_results.extend(np.array(each_log_probs[0]))

        # slice result 
        infer_results = np.array(infer_results)
        seq_len = (infer_data[2] - 1) // 3 + 1

        start_pos = [0] * (infer_data[0].shape[0] + 1)
        for i in range(infer_data[0].shape[0]):
            start_pos[i + 1] = start_pos[i] + seq_len[i][0]
        probs_split = [
            infer_results[start_pos[i]:start_pos[i + 1]]
            for i in range(0, infer_data[0].shape[0])
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
        print(results)
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
            for i in range(self._num_conv_layers):
                adapted_feeding_dict["conv%d_index_range" %i] = \
                        len(adapted_feeding_dict)
        elif isinstance(feeding_dict, list):
            adapted_feeding_dict.append("sequence_offset")
            adapted_feeding_dict.append("sequence_length")
            for i in range(self._num_conv_layers):
                adapted_feeding_dict.append("conv%d_index_range" % i)
        else:
            raise ValueError("Type of feeding_dict is %s, not supported." %
                             type(feeding_dict))

        return adapted_feeding_dict
