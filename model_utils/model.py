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

from utils.error_rate import char_errors, word_errors, cer, wer


class DeepSpeech2Trainer(Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)

    def compute_losses(self, inputs, outputs):
        _, texts, _, texts_len = inputs
        logits, _, logits_len = outputs
        loss = self.criterion(logits, texts, logits_len, texts_len)
        return loss

    def train_batch(self, batch_data):
        start = time.time()
        self.model.train()

        audio, text, audio_len, text_len = batch_data
        outputs = self.model(audio, text, audio_len, text_len)
        loss = self.compute_losses(batch_data, outputs)

        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

        iteration_time = time.time() - start

        losses_np = {
            'train_loss': float(loss),
            'train_loss_div_batchsize':
            float(loss) / self.config.data.batch_size
        }
        msg = "Train: Rank: {}, ".format(dist.get_rank())
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)
        msg += "time: {:>.3f}s, ".format(iteration_time)
        msg += ', '.join('{}: {:>.6f}'.format(k, v)
                         for k, v in losses_np.items())
        self.logger.info(msg)

        if dist.get_rank() == 0 and self.visualizer:
            for k, v in losses_np.items():
                self.visualizer.add_scalar("train/{}".format(k), v,
                                           self.iteration)

    def new_epoch(self):
        """Reset the train loader and increment ``epoch``.
        """
        if self.parallel:
            # batch sampler epoch start from 0
            self.train_loader.batch_sampler.set_epoch(self.epoch)
        self.epoch += 1

    def train(self):
        """The training process.
        
        It includes forward/backward/update and periodical validation and 
        saving.
        """
        self.logger.info(
            f"Train Total Examples: {len(self.train_loader.dataset)}")
        self.new_epoch()
        while self.epoch <= self.config.training.n_epoch:
            for batch in self.train_loader:
                self.iteration += 1
                self.train_batch(batch)

                # if self.iteration % self.config.training.valid_interval == 0:
                #     self.valid()

                # if self.iteration % self.config.training.save_interval == 0:
                #     self.save()

            self.valid()
            self.save()
            self.lr_scheduler.step()
            self.new_epoch()

    def compute_metrics(self, inputs, outputs):
        pass

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        self.logger.info(
            f"Valid Total Examples: {len(self.valid_loader.dataset)}")
        self.model.eval()
        valid_losses = defaultdict(list)
        for i, batch in enumerate(self.valid_loader):
            audio, text, audio_len, text_len = batch
            outputs = self.model(audio, text, audio_len, text_len)
            loss = self.compute_losses(batch, outputs)
            metrics = self.compute_metrics(batch, outputs)

            valid_losses['val_loss'].append(float(loss))
            valid_losses['val_loss_div_batchsize'].append(
                float(loss) / self.config.data.batch_size)

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

        for n, p in model.named_parameters():
            self.logger.info(
                f"param: {n}: shape: {p.shape} stop_grad: {p.stop_gradient}")

        grad_clip = paddle.nn.ClipGradByGlobalNorm(
            config.training.global_grad_clip)

        # optimizer = paddle.optimizer.Adam(
        #     learning_rate=config.training.lr,
        #     parameters=model.parameters(),
        #     weight_decay=paddle.regularizer.L2Decay(
        #         config.training.weight_decay),
        #     grad_clip=grad_clip)

        #learning_rate=fluid.layers.exponential_decay(
        #    learning_rate=learning_rate,
        #    decay_steps=num_samples / batch_size / dev_count,
        #    decay_rate=0.83,
        #    staircase=True),

        lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=config.training.lr, gamma=0.83, verbose=True)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            grad_clip=grad_clip)

        criterion = DeepSpeech2Loss(self.train_loader.dataset.vocab_size)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.logger.info("Setup model/optimizer/lr_scheduler/criterion!")

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


class DeepSpeech2Tester(DeepSpeech2Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)

    def id2token(self, texts, texts_len, vocab_list):
        trans = []
        for text, n in zip(texts, texts_len):
            n = n.numpy().item()
            ids = text[:n]
            trans.append(''.join([vocab_list[i] for i in ids]))
        return np.array(trans)

    def compute_metrics(self, inputs, outputs):
        cfg = self.config.decoding

        _, texts, _, texts_len = inputs
        logits, probs, logits_len = outputs

        errors_sum, len_refs, num_ins = 0.0, 0, 0
        errors_func = char_errors if cfg.error_rate_type == 'cer' else word_errors
        error_rate_func = cer if cfg.error_rate_type == 'cer' else wer

        vocab_list = self.test_loader.dataset.vocab_list
        target_transcripts = self.id2token(texts, texts_len, vocab_list)
        result_transcripts = self.model.decode_probs(
            probs.numpy(),
            vocab_list,
            decoding_method=cfg.decoding_method,
            lang_model_path=cfg.lang_model_path,
            beam_alpha=cfg.alpha,
            beam_beta=cfg.beta,
            beam_size=cfg.beam_size,
            cutoff_prob=cfg.cutoff_prob,
            cutoff_top_n=cfg.cutoff_top_n,
            num_processes=cfg.num_proc_bsearch)

        for target, result in zip(target_transcripts, result_transcripts):
            errors, len_ref = errors_func(target, result)
            errors_sum += errors
            len_refs += len_ref
            num_ins += 1
            self.logger.info(
                "\nTarget Transcription: %s\nOutput Transcription: %s" %
                (target, result))
            self.logger.info("Current error rate [%s] = %f" % (
                cfg.error_rate_type, error_rate_func(target, result)))

        return dict(
            errors_sum=errors_sum,
            len_refs=len_refs,
            num_ins=num_ins,
            error_rate=errors_sum / len_refs,
            error_rate_type=cfg.error_rate_type)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        self.logger.info(
            f"Test Total Examples: {len(self.test_loader.dataset)}")
        self.model.eval()
        losses = defaultdict(list)
        cfg = self.config
        # decoders only accept string encoded in utf-8
        vocab_list = self.test_loader.dataset.vocab_list
        self.model.init_decode(
            beam_alpha=cfg.decoding.alpha,
            beam_beta=cfg.decoding.beta,
            lang_model_path=cfg.decoding.lang_model_path,
            vocab_list=vocab_list,
            decoding_method=cfg.decoding.decoding_method)

        error_rate_type = None
        errors_sum, len_refs, num_ins = 0.0, 0, 0

        for i, batch in enumerate(self.test_loader):
            audio, text, audio_len, text_len = batch
            outputs = self.model.predict(audio, audio_len)
            loss = self.compute_losses(batch, outputs)
            losses['test_loss'].append(float(loss))

            metrics = self.compute_metrics(batch, outputs)
            errors_sum += metrics['errors_sum']
            len_refs += metrics['len_refs']
            num_ins += metrics['num_ins']
            error_rate_type = metrics['error_rate_type']
            self.logger.info("Error rate [%s] (%d/?) = %f" %
                             (error_rate_type, num_ins, errors_sum / len_refs))

        # write visual log
        losses = {k: np.mean(v) for k, v in losses.items()}

        # logging
        msg = "Test: "
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)
        msg += ', '.join('{}: {:>.6f}'.format(k, v) for k, v in losses.items())
        msg += ", Final error rate [%s] (%d/%d) = %f" % (
            error_rate_type, num_ins, num_ins, errors_sum / len_refs)
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
            batch_size=config.decoding.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)
        self.logger.info("Setup test Dataloader!")
