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
"""Contains wav2vec2 model."""
import json
import os
import time
from collections import defaultdict
from collections import OrderedDict
from contextlib import nullcontext

import jsonlines
import numpy as np
import paddle
from paddle import distributed as dist

from paddlespeech.s2t.frontend.featurizer import TextFeaturizer
from paddlespeech.s2t.io.dataloader import DataLoaderFactory
from paddlespeech.s2t.models.wav2vec2.processing.speech_augmentation import TimeDomainSpecAugment
from paddlespeech.s2t.models.wav2vec2.wav2vec2_ASR import Wav2vec2ASR
from paddlespeech.s2t.training.optimizer import OptimizerFactory
from paddlespeech.s2t.training.reporter import ObsScope
from paddlespeech.s2t.training.reporter import report
from paddlespeech.s2t.training.scheduler import LRSchedulerFactory
from paddlespeech.s2t.training.timer import Timer
from paddlespeech.s2t.training.trainer import Trainer
from paddlespeech.s2t.utils import error_rate
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils import mp_tools
from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.utility import UpdateConfig

logger = Log(__name__).getlog()


class Wav2Vec2ASRTrainer(Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.avg_train_loss = 0

    def update_average(self, batch_index, loss, avg_loss):
        """Update running average of the loss.
        Arguments
        ---------
        loss : paddle.tensor
            detached loss, a single float value.
        avg_loss : float
            current running average.
        Returns
        -------
        avg_loss : float
            The average loss.
        """
        if paddle.isfinite(loss):
            avg_loss -= avg_loss / (batch_index + 1)
            avg_loss += float(loss) / (batch_index + 1)
        return avg_loss

    def train_batch(self, batch_index, batch, msg):
        train_conf = self.config
        start = time.time()

        # forward
        utt, wav, wavs_lens, target, target_lens = batch
        wavs_lens_rate = wavs_lens / wav.shape[1]
        target_lens_rate = target_lens / target.shape[1]
        wav = wav[:, :, 0]
        wav = self.speech_augmentation(wav, wavs_lens_rate)
        loss = self.model(wav, wavs_lens_rate, target, target_lens_rate)
        # loss div by `batch_size * accum_grad`
        loss /= train_conf.accum_grad

        self.avg_train_loss = self.update_average(batch_index, loss,
                                                  self.avg_train_loss)

        # loss backward
        if (batch_index + 1) % train_conf.accum_grad != 0:
            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            # When using cpu w/o DDP, model does not have `no_sync`
            context = self.model.no_sync if (hasattr(self.model, "no_sync") and
                                             self.parallel) else nullcontext
        else:
            # Used for single gpu training and DDP gradient synchronization
            # processes.
            context = nullcontext
        with context():
            loss.backward()
            layer_tools.print_grads(self.model, print_func=None)

        # optimizer step old
        if (batch_index + 1) % train_conf.accum_grad == 0:
            self.optimizer.step()
            self.optimizer.clear_grad()
            self.lr_scheduler.step()
            self.iteration += 1

        losses_np = {'loss': float(self.avg_train_loss) * train_conf.accum_grad}
        iteration_time = time.time() - start
        for k, v in losses_np.items():
            report(k, v)
        report("batch_size", self.config.batch_size)
        report("accum", train_conf.accum_grad)
        report("step_cost", iteration_time)

        if (batch_index + 1) % train_conf.accum_grad == 0:
            if dist.get_rank() == 0 and self.visualizer:
                losses_np_v = losses_np.copy()
                losses_np_v.update({"lr": self.lr_scheduler()})
                for key, val in losses_np_v.items():
                    self.visualizer.add_scalar(
                        tag='train/' + key, value=val, step=self.iteration - 1)

    @paddle.no_grad()
    def valid(self):
        self.model.eval()
        if not self.use_streamdata:
            logger.info(
                f"Valid Total Examples: {len(self.valid_loader.dataset)}")
        valid_losses = defaultdict(list)
        num_seen_utts = 1
        total_loss = 0.0
        for i, batch in enumerate(self.valid_loader):
            utt, wav, wavs_lens, target, target_lens = batch
            wavs_lens_rate = wavs_lens / wav.shape[1]
            target_lens_rate = target_lens / target.shape[1]
            wav = wav[:, :, 0]
            loss = self.model(wav, wavs_lens_rate, target, target_lens_rate)

            if paddle.isfinite(loss):
                num_utts = batch[1].shape[0]
                num_seen_utts += num_utts
                total_loss += float(loss) * num_utts
                valid_losses['val_loss'].append(float(loss))

            if (i + 1) % self.config.log_interval == 0:
                valid_dump = {k: np.mean(v) for k, v in valid_losses.items()}
                valid_dump['val_history_loss'] = total_loss / num_seen_utts

                # logging
                msg = f"Valid: Rank: {dist.get_rank()}, "
                msg += "epoch: {}, ".format(self.epoch)
                msg += "step: {}, ".format(self.iteration)
                if not self.use_streamdata:
                    msg += "batch: {}/{}, ".format(i + 1,
                                                   len(self.valid_loader))
                msg += ', '.join('{}: {:>.6f}'.format(k, v)
                                 for k, v in valid_dump.items())
                logger.info(msg)

        logger.info('Rank {} Val info val_loss {}'.format(
            dist.get_rank(), total_loss / num_seen_utts))
        return total_loss, num_seen_utts

    def do_train(self):
        """The training process control by step."""
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        # script_model = paddle.jit.to_static(self.model)
        # script_model_path = str(self.checkpoint_dir / 'init')
        # paddle.jit.save(script_model, script_model_path)

        self.before_train()

        if not self.use_streamdata:
            logger.info(
                f"Train Total Examples: {len(self.train_loader.dataset)}")
        while self.epoch < self.config.n_epoch:
            with Timer("Epoch-Train Time Cost: {}"):
                self.model.train()
                try:
                    data_start_time = time.time()
                    for batch_index, batch in enumerate(self.train_loader):
                        dataload_time = time.time() - data_start_time
                        msg = "Train:"
                        observation = OrderedDict()
                        with ObsScope(observation):
                            report("Rank", dist.get_rank())
                            report("epoch", self.epoch)
                            report('step', self.iteration)
                            report("lr", self.lr_scheduler())
                            self.train_batch(batch_index, batch, msg)
                            self.after_train_batch()
                            report('iter', batch_index + 1)
                            if not self.use_streamdata:
                                report('total', len(self.train_loader))
                            report('reader_cost', dataload_time)
                        observation['batch_cost'] = observation[
                            'reader_cost'] + observation['step_cost']
                        observation['samples'] = observation['batch_size']
                        observation['ips,samples/s'] = observation[
                            'batch_size'] / observation['batch_cost']
                        for k, v in observation.items():
                            msg += f" {k.split(',')[0]}: "
                            msg += f"{v:>.8f}" if isinstance(v,
                                                             float) else f"{v}"
                            msg += f" {k.split(',')[1]}" if len(
                                k.split(',')) == 2 else ""
                            msg += ","
                        msg = msg[:-1]  # remove the last ","
                        if (batch_index + 1) % self.config.log_interval == 0:
                            logger.info(msg)
                        data_start_time = time.time()
                except Exception as e:
                    logger.error(e)
                    raise e
            with Timer("Eval Time Cost: {}"):
                total_loss, num_seen_utts = self.valid()
                if dist.get_world_size() > 1:
                    num_seen_utts = paddle.to_tensor(num_seen_utts)
                    # the default operator in all_reduce function is sum.
                    dist.all_reduce(num_seen_utts)
                    total_loss = paddle.to_tensor(total_loss)
                    dist.all_reduce(total_loss)
                    cv_loss = total_loss / num_seen_utts
                    cv_loss = float(cv_loss)
                else:
                    cv_loss = total_loss / num_seen_utts

            logger.info(
                'Epoch {} Val info val_loss {}'.format(self.epoch, cv_loss))
            if self.visualizer:
                self.visualizer.add_scalar(
                    tag='eval/cv_loss', value=cv_loss, step=self.epoch)
                self.visualizer.add_scalar(
                    tag='eval/lr', value=self.lr_scheduler(), step=self.epoch)

            self.save(tag=self.epoch, infos={'val_loss': cv_loss})
            self.new_epoch()

    def setup_dataloader(self):
        config = self.config.clone()
        self.use_streamdata = config.get("use_stream_data", False)
        if self.train:
            self.train_loader = DataLoaderFactory.get_dataloader(
                'train', config, self.args)
            self.valid_loader = DataLoaderFactory.get_dataloader(
                'valid', config, self.args)
            logger.info("Setup train/valid Dataloader!")
        else:
            decode_batch_size = config.get('decode', dict()).get(
                'decode_batch_size', 1)
            self.test_loader = DataLoaderFactory.get_dataloader('test', config,
                                                                self.args)
            self.align_loader = DataLoaderFactory.get_dataloader(
                'align', config, self.args)
            logger.info("Setup test/align Dataloader!")

    def setup_model(self):
        config = self.config
        model_conf = config

        with UpdateConfig(model_conf):
            if self.train:
                model_conf.input_dim = self.train_loader.feat_dim
                model_conf.output_dim = self.train_loader.vocab_size
            else:
                model_conf.input_dim = self.test_loader.feat_dim
                model_conf.output_dim = self.test_loader.vocab_size

        model = Wav2vec2ASR.from_config(model_conf)

        if self.parallel:
            model = paddle.DataParallel(model, find_unused_parameters=True)

        logger.info(f"{model}")
        layer_tools.print_params(model, logger.info)
        self.model = model
        logger.info("Setup model!")

        # setup speech augmentation for wav2vec2
        self.speech_augmentation = TimeDomainSpecAugment()

        if not self.train:
            return

        train_config = config
        optim_type = train_config.model_optim
        optim_conf = train_config.model_optim_conf
        scheduler_type = train_config.scheduler
        scheduler_conf = train_config.scheduler_conf

        scheduler_args = {
            "learning_rate": optim_conf.lr,
            "verbose": False,
            "warmup_steps": scheduler_conf.warmup_steps,
            "gamma": scheduler_conf.lr_decay,
            "d_model": model_conf.dnn_neurons,
        }
        lr_scheduler = LRSchedulerFactory.from_args(scheduler_type,
                                                    scheduler_args)

        def optimizer_args(
                config,
                parameters,
                lr_scheduler=None, ):
            train_config = config
            optim_type = train_config.model_optim
            optim_conf = train_config.model_optim_conf
            scheduler_type = train_config.scheduler
            scheduler_conf = train_config.scheduler_conf
            return {
                "grad_clip": train_config.global_grad_clip,
                "learning_rate": lr_scheduler
                if lr_scheduler else optim_conf.lr,
                "epsilon": optim_conf.epsilon,
                "rho": optim_conf.rho,
                "parameters": parameters,
                "beta1": 0.9 if optim_type == 'noam' else None,
                "beat2": 0.98 if optim_type == 'noam' else None,
            }

        optimzer_args = optimizer_args(config, model.parameters(), lr_scheduler)
        optimizer = OptimizerFactory.from_args(optim_type, optimzer_args)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        logger.info("Setup optimizer/lr_scheduler!")


class Wav2Vec2ASRTester(Wav2Vec2ASRTrainer):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.text_featurizer = TextFeaturizer(
            unit_type=config.unit_type, vocab=config.vocab_filepath)
        self.vocab_list = self.text_featurizer.vocab_list

    def id2token(self, texts, texts_len):
        """ ord() id to chr() chr """
        trans = []
        for text, n in zip(texts, texts_len):
            n = n.numpy().item()
            ids = text[:n]
            trans.append(self.text_featurizer.defeaturize(ids.numpy().tolist()))
        return trans

    def compute_metrics(self,
                        utts,
                        audio,
                        audio_len,
                        texts,
                        texts_len,
                        fout=None):
        decode_cfg = self.config.decode
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        errors_func = error_rate.char_errors if decode_cfg.error_rate_type == 'cer' else error_rate.word_errors
        error_rate_func = error_rate.cer if decode_cfg.error_rate_type == 'cer' else error_rate.wer

        start_time = time.time()
        target_transcripts = self.id2token(texts, texts_len)
        result_transcripts, result_tokenids = self.model.decode(
            audio,
            text_feature=self.text_featurizer,
            decoding_method=decode_cfg.decoding_method,
            beam_size=decode_cfg.beam_size)
        decode_time = time.time() - start_time

        for utt, target, result, rec_tids in zip(
                utts, target_transcripts, result_transcripts, result_tokenids):
            errors, len_ref = errors_func(target, result)
            errors_sum += errors
            len_refs += len_ref
            num_ins += 1
            if fout:
                fout.write({
                    "utt": utt,
                    "refs": [target],
                    "hyps": [result],
                    "hyps_tokenid": [rec_tids],
                })
            logger.info(f"Utt: {utt}")
            logger.info(f"Ref: {target}")
            logger.info(f"Hyp: {result}")
            logger.info("One example error rate [%s] = %f" % (
                decode_cfg.error_rate_type, error_rate_func(target, result)))

        return dict(
            errors_sum=errors_sum,
            len_refs=len_refs,
            num_ins=num_ins,  # num examples
            error_rate=errors_sum / len_refs,
            error_rate_type=decode_cfg.error_rate_type,
            num_frames=audio_len.sum().numpy().item(),
            decode_time=decode_time)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def test(self):
        logger.info(f"Test Total Examples: {len(self.test_loader.dataset)}")
        self.model.eval()

        error_rate_type = None
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        num_frames = 0.0
        num_time = 0.0
        # Initialized the decoder in model
        decode_cfg = self.config.decode
        vocab_list = self.vocab_list
        decode_batch_size = decode_cfg.decode_batch_size

        with jsonlines.open(self.args.result_file, 'w') as fout:
            for i, batch in enumerate(self.test_loader):
                metrics = self.compute_metrics(*batch, fout=fout)
                num_frames += metrics['num_frames']
                num_time += metrics["decode_time"]
                errors_sum += metrics['errors_sum']
                len_refs += metrics['len_refs']
                num_ins += metrics['num_ins']
                error_rate_type = metrics['error_rate_type']
                rtf = num_time / (num_frames)
                logger.info(
                    "RTF: %f, Error rate [%s] (%d/?) = %f" %
                    (rtf, error_rate_type, num_ins, errors_sum / len_refs))

        # logging
        msg = "Test: "
        msg += "epoch: {}, ".format(self.epoch)
        msg += "step: {}, ".format(self.iteration)
        msg += "Final error rate [%s] (%d/%d) = %f" % (
            error_rate_type, num_ins, num_ins, errors_sum / len_refs)
        logger.info(msg)

        err_meta_path = os.path.splitext(self.args.result_file)[0] + '.err'
        err_type_str = "{}".format(error_rate_type)
        with open(err_meta_path, 'w') as f:
            data = json.dumps({
                "epoch":
                self.epoch,
                "step":
                self.iteration,
                "rtf":
                rtf,
                error_rate_type:
                errors_sum / len_refs,
                "dataset_hour": (num_frames) / 1000.0 / 3600.0,
                "process_hour":
                num_time / 1000.0 / 3600.0,
                "num_examples":
                num_ins,
                "err_sum":
                errors_sum,
                "ref_len":
                len_refs,
                "decode_method":
                self.config.decode.decoding_method,
            })
            f.write(data + '\n')
