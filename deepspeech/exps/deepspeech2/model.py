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
import logging
import numpy as np
from collections import defaultdict
from functools import partial
from pathlib import Path

import paddle
from paddle import distributed as dist
from paddle.io import DataLoader

from deepspeech.training import Trainer
from deepspeech.training.gradclip import ClipGradByGlobalNormWithLog

from deepspeech.utils import mp_tools
from deepspeech.utils import layer_tools
from deepspeech.utils import error_rate

from deepspeech.io.collator import SpeechCollator
from deepspeech.io.sampler import SortagradDistributedBatchSampler
from deepspeech.io.sampler import SortagradBatchSampler
from deepspeech.io.dataset import ManifestDataset

from deepspeech.modules.loss import CTCLoss
from deepspeech.models.deepspeech2 import DeepSpeech2Model
from deepspeech.models.deepspeech2 import DeepSpeech2InferModel

logger = logging.getLogger(__name__)


class DeepSpeech2Trainer(Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)

    def train_batch(self, batch_data):
        start = time.time()
        self.model.train()
        loss = self.model(*batch_data)
        loss.backward()
        layer_tools.print_grads(self.model, print_func=None)
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

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        self.logger.info(
            f"Valid Total Examples: {len(self.valid_loader.dataset)}")
        self.model.eval()
        valid_losses = defaultdict(list)
        for i, batch in enumerate(self.valid_loader):
            loss = self.model(*batch)

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
        model = DeepSpeech2Model(
            feat_size=self.train_loader.dataset.feature_size,
            dict_size=self.train_loader.dataset.vocab_size,
            num_conv_layers=config.model.num_conv_layers,
            num_rnn_layers=config.model.num_rnn_layers,
            rnn_size=config.model.rnn_layer_size,
            use_gru=config.model.use_gru,
            share_rnn_weights=config.model.share_rnn_weights)

        if self.parallel:
            model = paddle.DataParallel(model)

        layer_tools.print_params(model, self.logger.info)

        grad_clip = ClipGradByGlobalNormWithLog(
            config.training.global_grad_clip)
        lr_scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=config.training.lr,
            gamma=config.training.lr_decay,
            verbose=True)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=paddle.regularizer.L2Decay(
                config.training.weight_decay),
            grad_clip=grad_clip)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger.info("Setup model/optimizer/lr_scheduler!")

    def setup_dataloader(self):
        config = self.config

        train_dataset = ManifestDataset(
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

        dev_dataset = ManifestDataset(
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
            batch_sampler = SortagradDistributedBatchSampler(
                train_dataset,
                batch_size=config.data.batch_size,
                num_replicas=None,
                rank=None,
                shuffle=True,
                drop_last=True,
                sortagrad=config.data.sortagrad,
                shuffle_method=config.data.shuffle_method)
        else:
            batch_sampler = SortagradBatchSampler(
                train_dataset,
                shuffle=True,
                batch_size=config.data.batch_size,
                drop_last=True,
                sortagrad=config.data.sortagrad,
                shuffle_method=config.data.shuffle_method)

        collate_fn = SpeechCollator(is_training=True)
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

    def ordid2token(self, texts, texts_len):
        """ ord() id to chr() chr """
        trans = []
        for text, n in zip(texts, texts_len):
            n = n.numpy().item()
            ids = text[:n]
            trans.append(''.join([chr(i) for i in ids]))
        return trans

    def compute_metrics(self, audio, texts, audio_len, texts_len):
        cfg = self.config.decoding
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        errors_func = error_rate.char_errors if cfg.error_rate_type == 'cer' else error_rate.word_errors
        error_rate_func = error_rate.cer if cfg.error_rate_type == 'cer' else error_rate.wer

        vocab_list = self.test_loader.dataset.vocab_list

        target_transcripts = self.ordid2token(texts, texts_len)
        result_transcripts = self.model.decode(
            audio,
            audio_len,
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
        cfg = self.config
        error_rate_type = None
        errors_sum, len_refs, num_ins = 0.0, 0, 0

        for i, batch in enumerate(self.test_loader):
            metrics = self.compute_metrics(*batch)
            errors_sum += metrics['errors_sum']
            len_refs += metrics['len_refs']
            num_ins += metrics['num_ins']
            error_rate_type = metrics['error_rate_type']
            self.logger.info("Error rate [%s] (%d/?) = %f" %
                             (error_rate_type, num_ins, errors_sum / len_refs))

        # logging
        msg = "Test: "
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)
        msg += ", Final error rate [%s] (%d/%d) = %f" % (
            error_rate_type, num_ins, num_ins, errors_sum / len_refs)
        self.logger.info(msg)

    def run_test(self):
        self.resume_or_load()
        try:
            self.test()
        except KeyboardInterrupt:
            exit(-1)

    def export(self):
        infer_model = DeepSpeech2InferModel.from_pretrained(
            self.test_loader.dataset, self.config, self.args.checkpoint_path)
        infer_model.eval()
        feat_dim = self.test_loader.dataset.feature_size
        static_model = paddle.jit.to_static(
            infer_model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, feat_dim, None],
                    dtype='float32'),  # audio, [B,D,T]
                paddle.static.InputSpec(shape=[None],
                                        dtype='int64'),  # audio_length, [B]
            ])
        logger.info(f"Export code: {static_model.forward.code}")
        paddle.jit.save(static_model, self.args.export_path)

    def run_export(self):
        try:
            self.export()
        except KeyboardInterrupt:
            exit(-1)

    def setup(self):
        """Setup the experiment.
        """
        paddle.set_device(self.args.device)

        self.setup_output_dir()
        self.setup_checkpointer()
        self.setup_logger()

        self.setup_dataloader()
        self.setup_model()

        self.iteration = 0
        self.epoch = 0

    def setup_model(self):
        config = self.config
        model = DeepSpeech2Model(
            feat_size=self.test_loader.dataset.feature_size,
            dict_size=self.test_loader.dataset.vocab_size,
            num_conv_layers=config.model.num_conv_layers,
            num_rnn_layers=config.model.num_rnn_layers,
            rnn_size=config.model.rnn_layer_size,
            use_gru=config.model.use_gru,
            share_rnn_weights=config.model.share_rnn_weights)
        self.model = model
        self.logger.info("Setup model!")

    def setup_dataloader(self):
        config = self.config
        # return raw text
        test_dataset = ManifestDataset(
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
            keep_transcription_text=True)

        # return text ord id
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.decoding.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=SpeechCollator(is_training=False))
        self.logger.info("Setup test Dataloader!")

    def setup_output_dir(self):
        """Create a directory used for output.
        """
        # output dir
        if self.args.output:
            output_dir = Path(self.args.output).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(
                self.args.checkpoint_path).expanduser().parent.parent
            output_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir

    def setup_logger(self):
        """Initialize a text logger to log the experiment.
        
        Each process has its own text logger. The logging message is write to 
        the standard output and a text file named ``worker_n.log`` in the 
        output directory, where ``n`` means the rank of the process. 
        """
        format = '[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
        formatter = logging.Formatter(fmt=format, datefmt='%Y/%m/%d %H:%M:%S')

        logger.setLevel("INFO")

        # global logger
        stdout = True
        save_path = ""
        logging.basicConfig(
            level=logging.DEBUG if stdout else logging.INFO,
            format=format,
            datefmt='%Y/%m/%d %H:%M:%S',
            filename=save_path if not stdout else None)
        self.logger = logger
