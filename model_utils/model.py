# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains DeepSpeech2 model."""

import sys
import os
import time
import logging
import gzip
import copy
import inspect
import collections
import multiprocessing
import numpy as np
from distutils.dir_util import mkpath

import paddle.fluid as fluid

from training import Trainer
from model_utils.network import DeepSpeech2
from model_utils.network import DeepSpeech2Loss

from decoders.swig_wrapper import Scorer
from decoders.swig_wrapper import ctc_greedy_decoder
from decoders.swig_wrapper import ctc_beam_search_decoder_batch

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')


class SpeechCollator():
    def __init__(self, padding_to=-1):
        """
        Padding audio features with zeros to make them have the same shape (or
        a user-defined shape) within one bach.

        If ``padding_to`` is -1, the maximun shape in the batch will be used
        as the target shape for padding. Otherwise, `padding_to` will be the
        target shape (only refers to the second axis).
        """
        self._padding_to = padding_to

    def __call__(self, batch):
        new_batch = []
        # get target shape
        max_length = max([audio.shape[1] for audio, _ in batch])
        if self._padding_to != -1:
            if self._padding_to < max_length:
                raise ValueError("If padding_to is not -1, it should be larger "
                                 "than any instance's shape in the batch")
            max_length = self._padding_to
        max_text_length = max([len(text) for _, text in batch])
        # padding
        padded_audios = []
        audio_lens = []
        texts, text_lens = [], []
        for audio, text in batch:
            # audio
            padded_audio = np.zeros([audio.shape[0], max_length])
            padded_audio[:, :audio.shape[1]] = audio
            padded_audios.append(padded_audio)
            audio_lens.append(audio.shape[1])
            # text
            padded_text = np.zeros([max_text_length])
            padded_text[:len(text)] = text
            texts.append(padded_text)
            text_lens.append(len(text))

        padded_audios = np.array(padded_audios).astype('float32')
        audio_lens = np.array(audio_lens).astype('int64')
        texts = np.array(texts).astype('int32')
        text_lens = np.array(text_lens).astype('int64')
        return padded_audios, texts, audio_lens, text_lens


class DeepSpeech2Trainer(Trainer):
    def __init__(self):
        self._ext_scorer = None

    def setup_dataloader(self):
        config = self.config

        train_dataset = DeepSpeech2Dataset(
            config.data.train_manifest_path,
            config.data.vocab_filepath,
            config.data.mean_std_filepath,
            augmentation_config=config.data.augmentation_config,
            max_duration=config.data.max_duration,
            min_duration=config.data.min_duration,
            stride_ms=config.data.stride_ms,
            window_ms=config.data.window_ms,
            max_freq=config.data.max_freq,
            specgram_type=config.data.specgram_type,
            use_dB_normalization=config.data.use_dB_normalization,
            random_seed=config.data.random_seed,
            keep_transcription_text=False)

        dev_dataset = DeepSpeech2Dataset(
            config.data.dev_manifest_path,
            config.data.vocab_filepath,
            config.data.mean_std_filepath,
            augmentation_config=config.data.augmentation_config,
            max_duration=config.data.max_duration,
            min_duration=config.data.min_duration,
            stride_ms=config.data.stride_ms,
            window_ms=config.data.window_ms,
            max_freq=config.data.max_freq,
            specgram_type=config.data.specgram_type,
            use_dB_normalization=config.data.use_dB_normalization,
            random_seed=config.data.random_seed,
            keep_transcription_text=False)

        if self.parallel:
            batch_sampler = DeepSpeech2DistributedBatchSampler(
                train_dataset,
                batch_size=config.data.batch_size,
                num_replicas=None,
                rank=None,
                shuffle=True,
                drop_last=True,
                sortagrad=config.data.sortagrad,
                shuffle_method=config.data.shuffle_method)
        else:
            batch_sampler = DeepSpeech2BatchSampler(
                train_dataset,
                shuffle=True,
                batch_size=config.data.batch_size,
                drop_last=True,
                sortagrad=config.data.sortagrad,
                shuffle_method=config.data.shuffle_method)

        collate_fn = SpeechCollator()
        self.train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=config.data.num_workers, )
        self.valid_loader = DataLoader(
            dev_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)
        self.logger.info("Setup train/valid Dataloader!")

    def setup_model(self):
        config = self.config
        model = DeepSpeech2(
            feat_size=self.train_loader.feature_size,
            dict_size=self.train_loader.vocab_size,
            num_conv_layers=config.model.num_conv_layers,
            num_rnn_layers=config.model.num_rnn_layers,
            rnn_size=config.model.rnn_layer_size,
            share_rnn_weights=config.model.share_rnn_weights)

        if self.parallel:
            model = paddle.DataParallel(model)

        grad_clip = paddle.nn.ClipGradByGlobalNorm(config.training.grad_clip)

        optimizer = paddle.optimizer.Adam(
            learning_rate=config.training.lr,
            parameters=model.parameters(),
            weight_decay=paddle.regulaerizer.L2Decay(
                config.training.weight_decay),
            grad_clip=grad_clip, )

        criterion = DeepSpeech2Loss(self.train_loader.vocab_size)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger.info("Setup model/optimizer/criterion!")

    def compute_losses(self, inputs, outputs):
        pass

    def test(self, test_reader):
        '''Test the model.

        :param exe:The executor of program.
        :type exe: Executor
        :param test_program: The program of test.
        :type test_program: Program
        :param test_reader: Reader of test.
        :type test_reader: Reader
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
                        staircase=True),
                    grad_clip=fluid.clip.GradientClipByGlobalNorm(
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
        :rtype: List of str
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
        :type language_model_path: str|None
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
        :rtype: List of str
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
