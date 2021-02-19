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

import io
import sys
import os
import time
import numpy as np
from collections import defaultdict

import paddle
from paddle import distributed as dist
from paddle.io import DataLoader

from utils import mp_tools
from training import Trainer

from model_utils.network import DeepSpeech2
from model_utils.network import DeepSpeech2Loss

from data_utils.dataset import SpeechCollator
from data_utils.dataset import DeepSpeech2Dataset
from data_utils.dataset import DeepSpeech2DistributedBatchSampler
from data_utils.dataset import DeepSpeech2BatchSampler

from decoders.swig_wrapper import Scorer
from decoders.swig_wrapper import ctc_greedy_decoder
from decoders.swig_wrapper import ctc_beam_search_decoder_batch


class DeepSpeech2Trainer(Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)

    def compute_losses(self, inputs, outputs):
        _, texts, _, texts_len = inputs
        logits, _, logits_len = outputs
        loss = self.criterion(logits, texts, logits_len, texts_len)
        return loss

    def read_batch(self):
        """Read a batch from the train_loader.
        Returns
        -------
        List[Tensor]
            A batch.
        """
        try:
            batch = next(self.iterator)
        except StopIteration as e:
            raise e
        return batch

    def train_batch(self):
        start = time.time()
        batch = self.read_batch()
        data_loader_time = time.time() - start

        self.optimizer.clear_grad()
        self.model.train()
        audio, text, audio_len, text_len = batch
        outputs = self.model(audio, text, audio_len, text_len)
        loss = self.compute_losses(batch, outputs)
        loss.backward()
        self.optimizer.step()
        iteration_time = time.time() - start

        losses_np = {'train_loss': float(loss)}
        msg = "Train: Rank: {}, ".format(dist.get_rank())
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)

        msg += "time: {:>.3f}s/{:>.3f}s, ".format(data_loader_time,
                                                  iteration_time)
        msg += ', '.join('{}: {:>.6f}'.format(k, v)
                         for k, v in losses_np.items())
        self.logger.info(msg)

        if dist.get_rank() == 0 and self.visualizer:
            for k, v in losses_np.items():
                self.visualizer.add_scalar("train/{}".format(k), v,
                                           self.iteration)

    def train(self):
        """The training process.
        
        It includes forward/backward/update and periodical validation and 
        saving.
        """
        self.new_epoch()
        while self.epoch <= self.config.training.n_epoch:
            try:
                self.iteration += 1
                self.train_batch()

                if self.iteration % self.config.training.valid_interval == 0:
                    self.valid()

                if self.iteration % self.config.training.save_interval == 0:
                    self.save()
            except StopIteration:
                self.iteration -= 1  #epoch end, iteration ahead 1
                self.valid()
                self.save()
                self.new_epoch()

    def compute_metrics(self, inputs, outputs):
        pass

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        self.model.eval()
        valid_losses = defaultdict(list)
        for i, batch in enumerate(self.valid_loader):
            audio, text, audio_len, text_len = batch
            outputs = self.model(audio, text, audio_len, text_len)
            loss = self.compute_losses(batch, outputs)
            metrics = self.compute_metrics(batch, outputs)

            valid_losses['val_loss'].append(float(loss))

        # write visual log
        valid_losses = {k: np.mean(v) for k, v in valid_losses.items()}

        # logging
        msg = f"Valid: Rank: {dist.get_rank()}, "
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)
        msg += ', '.join('{}: {:>.6f}'.format(k, v)
                         for k, v in valid_losses.items())
        self.logger.info(msg)

        if self.visualizer:
            for k, v in valid_losses.items():
                self.visualizer.add_scalar("valid/{}".format(k), v,
                                           self.iteration)

    def setup_model(self):
        config = self.config
        model = DeepSpeech2(
            feat_size=self.train_loader.dataset.feature_size,
            dict_size=self.train_loader.dataset.vocab_size,
            num_conv_layers=config.model.num_conv_layers,
            num_rnn_layers=config.model.num_rnn_layers,
            rnn_size=config.model.rnn_layer_size,
            share_rnn_weights=config.model.share_rnn_weights)

        if self.parallel:
            model = paddle.DataParallel(model)

        grad_clip = paddle.nn.ClipGradByGlobalNorm(
            config.training.global_grad_clip)

        optimizer = paddle.optimizer.Adam(
            learning_rate=config.training.lr,
            parameters=model.parameters(),
            weight_decay=paddle.regularizer.L2Decay(
                config.training.weight_decay),
            grad_clip=grad_clip)

        criterion = DeepSpeech2Loss(self.train_loader.dataset.vocab_size)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger.info("Setup model/optimizer/criterion!")

    def setup_dataloader(self):
        config = self.config

        train_dataset = DeepSpeech2Dataset(
            config.data.train_manifest,
            config.data.vocab_filepath,
            config.data.mean_std_filepath,
            augmentation_config=io.open(
                config.data.augmentation_config, mode='r',
                encoding='utf8').read(),
            max_duration=config.data.max_duration,
            min_duration=config.data.min_duration,
            stride_ms=config.data.stride_ms,
            window_ms=config.data.window_ms,
            n_fft=config.data.n_fft,
            max_freq=config.data.max_freq,
            target_sample_rate=config.data.target_sample_rate,
            specgram_type=config.data.specgram_type,
            use_dB_normalization=config.data.use_dB_normalization,
            target_dB=config.data.target_dB,
            random_seed=config.data.random_seed,
            keep_transcription_text=False)

        dev_dataset = DeepSpeech2Dataset(
            config.data.dev_manifest,
            config.data.vocab_filepath,
            config.data.mean_std_filepath,
            augmentation_config="{}",
            max_duration=config.data.max_duration,
            min_duration=config.data.min_duration,
            stride_ms=config.data.stride_ms,
            window_ms=config.data.window_ms,
            n_fft=config.data.n_fft,
            max_freq=config.data.max_freq,
            target_sample_rate=config.data.target_sample_rate,
            specgram_type=config.data.specgram_type,
            use_dB_normalization=config.data.use_dB_normalization,
            target_dB=config.data.target_dB,
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


class DeepSpeech2Tester(Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)

    def compute_losses(self, inputs, outputs):
        _, texts, _, texts_len = inputs
        logits, _, logits_len = outputs
        loss = self.criterion(logits, texts, logits_len, texts_len)
        return loss

    def compute_metrics(self, inputs, outputs):
        _, texts, _, texts_len = inputs
        logits, _, logits_len = outputs
        pass

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        self.model.eval()
        losses = defaultdict(list)
        for i, batch in enumerate(self.test_loader):
            audio, text, audio_len, text_len = batch
            outputs = self.model.predict(audio, audio_len)
            loss = self.compute_losses(batch, outputs)
            metrics = self.compute_metrics(batch, outputs)

            losses['test_loss'].append(float(loss))

        # write visual log
        losses = {k: np.mean(v) for k, v in losses.items()}

        # logging
        msg = "Test: "
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)
        msg += ', '.join('{}: {:>.6f}'.format(k, v) for k, v in losses.items())
        self.logger.info(msg)

    def setup(self):
        """Setup the experiment.
        """
        paddle.set_device(self.args.device)
        if self.parallel:
            self.init_parallel()

        self.setup_output_dir()
        self.setup_logger()
        self.setup_checkpointer()

        self.setup_dataloader()
        self.setup_model()

        self.iteration = 0
        self.epoch = 0

    def run_test(self):
        self.resume_or_load()
        try:
            self.test()
        except KeyboardInterrupt:
            exit(-1)
        finally:
            self.destory()

    def setup_model(self):
        config = self.config
        model = DeepSpeech2(
            feat_size=self.test_loader.dataset.feature_size,
            dict_size=self.test_loader.dataset.vocab_size,
            num_conv_layers=config.model.num_conv_layers,
            num_rnn_layers=config.model.num_rnn_layers,
            rnn_size=config.model.rnn_layer_size,
            share_rnn_weights=config.model.share_rnn_weights)

        if self.parallel:
            model = paddle.DataParallel(model)

        criterion = DeepSpeech2Loss(self.test_loader.dataset.vocab_size)

        self.model = model
        self.criterion = criterion
        self.logger.info("Setup model/criterion!")

    def setup_dataloader(self):
        config = self.config
        test_dataset = DeepSpeech2Dataset(
            config.data.test_manifest,
            config.data.vocab_filepath,
            config.data.mean_std_filepath,
            augmentation_config="{}",
            max_duration=config.data.max_duration,
            min_duration=config.data.min_duration,
            stride_ms=config.data.stride_ms,
            window_ms=config.data.window_ms,
            n_fft=config.data.n_fft,
            max_freq=config.data.max_freq,
            target_sample_rate=config.data.target_sample_rate,
            specgram_type=config.data.specgram_type,
            use_dB_normalization=config.data.use_dB_normalization,
            target_dB=config.data.target_dB,
            random_seed=config.data.random_seed,
            keep_transcription_text=False)

        collate_fn = SpeechCollator()
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)
        self.logger.info("Setup test Dataloader!")
