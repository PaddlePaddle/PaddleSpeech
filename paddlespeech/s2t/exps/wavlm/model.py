# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
"""Contains wavlm model."""
import json
import math
import os
import re
import time
from collections import OrderedDict
from contextlib import nullcontext

import jsonlines
import numpy as np
import paddle
from hyperpyyaml import load_hyperpyyaml
from paddle import distributed as dist
from paddlenlp.transformers import AutoTokenizer

from paddlespeech.s2t.frontend.featurizer import TextFeaturizer
from paddlespeech.s2t.io.dataloader import DataLoaderFactory
from paddlespeech.s2t.io.speechbrain import data_pipeline
from paddlespeech.s2t.io.speechbrain import dataio
from paddlespeech.s2t.io.speechbrain import dataset
from paddlespeech.s2t.io.speechbrain.dataloader import make_dataloader
from paddlespeech.s2t.models.wav2vec2.processing.speech_augmentation import TimeDomainSpecAugment
from paddlespeech.s2t.models.wavlm.wavlm_asr import WavLMASR
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


def clip_grad_norm_(
        parameters,
        max_norm,
        norm_type=2.0,
        error_if_nonfinite=False, ):
    r"""Clips gradient norm of the iteratable parameters.

    Norms are calculated together on all gradients, just as they are
    connected into one vector. The gradient will be modified in place.

    This API can only run in dynamic graph mode, not static graph mode.

    Args:
        parameters (Iterable[paddle.Tensor] or paddle.Tensor): Tensors or a single Tensor
            that will be normalized gradients
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be `inf` for
            infinity norm.
        error_if_nonfinite (bool): if True, throw an error if the total
            norm of the gradients from :attr:`parameters` is `nan`,
            `inf`, or `-inf`.

    Returns:
        Total norm of the parameter gradients (treated as a single vector).
    Example:
        .. code-block:: python
            import paddle

            x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
            max_norm = float(5.0)
            linear = paddle.nn.Linear(in_features=10, out_features=10)
            out = linear(x)
            loss = paddle.mean(out)
            loss.backward()

            paddle.nn.utils.clip_grad_norm_(linear.parameters(), max_norm)

            sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters())
            sdg.step()
    """
    if not paddle.in_dynamic_mode():
        raise RuntimeError('this API can only run in dynamic mode.')

    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]

    support_norm_type = [float("inf"), 0, 1, 2]
    if norm_type not in support_norm_type:
        raise ValueError(f'norm_type only support {support_norm_type}')

    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return paddle.to_tensor(0.0)
    if norm_type == float("inf"):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = (norms[0]
                      if len(norms) == 1 else paddle.max(paddle.stack(norms)))
    else:
        total_norm = paddle.linalg.norm(
            paddle.stack(
                [paddle.linalg.norm(g.detach(), norm_type) for g in grads]),
            norm_type, )

    if error_if_nonfinite and paddle.logical_or(total_norm.isnan(),
                                                total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of {norm_type} order of the gradients from '
            '`parameters` is non-finite, so it cannot be clipped. In any case, '
            'disable this error and scale the gradient by non-finite norm, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: when the coef is clamped to 1, it is redundant to multiply the clamped coef, but this
    # avoids the `if clip_coef < 1:` condition.
    clip_coef_clamped = paddle.clip(clip_coef, max=1.0)
    with paddle.no_grad():
        for _, p in enumerate(parameters):
            g = p.grad
            if g is not None:
                p.grad = paddle.multiply(x=g, y=clip_coef_clamped)
    return total_norm


class WavLMASRTrainer(Trainer):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.avg_train_loss = 0.0
        self.loss_isfinite = True  # while flag is 'False', loss in Nan or inf, and can not be avg
        self.use_sb = True  # whether use speech brain dataloader

    def update_average(self, batch_index, loss):
        """Update running average of the loss.
        Arguments
        ---------
        batch_index : int
            current batch index
        loss : paddle.tensor
            detached loss, a single float value.
        """
        if math.isfinite(loss):
            self.avg_train_loss -= self.avg_train_loss / (batch_index + 1)
            self.avg_train_loss += loss / (batch_index + 1)
        else:
            self.loss_isfinite = False
            logger.info('loss:{} in Nan or inf, error'.format(loss))

    def before_train(self):
        from_scratch = self.resume_or_scratch()
        if from_scratch:
            # scratch: save init model, i.e. 0 epoch
            self.save(tag='init', infos=None)
        else:
            # resume: train next_epoch and next_iteration
            self.epoch += 1
            logger.info(
                f"Resume train: epoch {self.epoch }, step {self.iteration}!")

        self.maybe_batch_sampler_step()

    def train_batch(self, batch_index, batch, msg):
        train_conf = self.config
        start = time.time()

        # forward
        ## sb data pipeline
        if self.use_sb:
            wav, wavs_lens_rate = batch['sig']
            target, target_lens_rate = batch['tokens']
            target_lens = (target_lens_rate *
                           target.shape[1]).round().astype(paddle.int64)
        else:
            utt, wav, wavs_lens, target, target_lens = batch
            wavs_lens_rate = wavs_lens / wav.shape[1]
            wav = wav[:, :, 0]

        if hasattr(train_conf, 'audio_augment'):
            wav = self.speech_augmentation(wav, wavs_lens_rate)
        loss = self.model(wav, wavs_lens_rate, target, target_lens)

        # loss div by `batch_size * accum_grad`
        loss /= train_conf.accum_grad
        # update self.avg_train_loss
        self.update_average(batch_index, float(loss))

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

        # NOTE: the code below asserted that the backward() is problematic, and as more steps are accumulated, the output from wavlm alone will be the same for all frames
        # optimizer step old
        if (batch_index + 1) % train_conf.accum_grad == 0:
            #do global grad clip
            if train_conf.global_grad_clip != 0:
                clip_grad_norm_(self.model.parameters(),
                                train_conf.global_grad_clip)
            self.model_optimizer.step()
            self.model_optimizer.clear_grad()
            if not train_conf.freeze_wavlm:
                self.wavlm_optimizer.step()
                self.wavlm_optimizer.clear_grad()
            if self.config.model_scheduler != 'newbobscheduler':
                self.model_lr_scheduler.step()
            if self.config.wavlm_scheduler != 'newbobscheduler':
                if not train_conf.freeze_wavlm:
                    self.wavlm_lr_scheduler.step()
            self.iteration += 1

        losses_np = {'loss': self.avg_train_loss * train_conf.accum_grad}
        iteration_time = time.time() - start
        for k, v in losses_np.items():
            report(k, v)
        report("loss_whitoutavg", float(loss))
        report("batch_size", self.config.batch_size)
        report("accum", train_conf.accum_grad)
        report("step_cost", iteration_time)

        if (batch_index + 1) % train_conf.accum_grad == 0:
            if dist.get_rank() == 0 and self.visualizer:
                losses_np_v = losses_np.copy()
                losses_np_v.update({
                    "model_lr": self.model_lr_scheduler(),
                    "wavlm_lr": self.wavlm_lr_scheduler()
                })
                for key, val in losses_np_v.items():
                    self.visualizer.add_scalar(
                        tag='train/' + key, value=val, step=self.iteration - 1)

    @paddle.no_grad()
    def valid(self):
        self.model.eval()
        if not self.use_streamdata:
            logger.info(
                f"Valid Total Examples: {len(self.valid_loader.dataset)}")
        valid_losses = {}
        step = 0
        total_loss = 0.0
        num_seen_utts = 1  # use update_average and no need for num_seen_utts here
        for i, batch in enumerate(self.valid_loader):
            if self.use_sb:
                wav, wavs_lens_rate = batch['sig']
                target, target_lens_rate = batch['tokens']
                target_lens = (target_lens_rate *
                               target.shape[1]).round().astype(paddle.int64)
            else:
                utt, wav, wavs_lens, target, target_lens = batch
                wavs_lens_rate = wavs_lens / wav.shape[1]
                wav = wav[:, :, 0]

            loss = self.model(wav, wavs_lens_rate, target, target_lens)
            # use update_average
            total_loss -= total_loss / (step + 1)
            total_loss += loss / (step + 1)

            if math.isfinite(float(loss)):
                step += 1
                valid_losses['val_loss'] = float(loss)
            else:
                logger.info('loss:{} in Nan or inf, error'.format(float(loss)))

            if (i + 1) % self.config.log_interval == 0:
                valid_losses['val_history_loss'] = float(total_loss)

                # logging
                msg = f"Valid: Rank: {dist.get_rank()}, "
                msg += "epoch: {}, ".format(self.epoch)
                msg += "step: {}, ".format(self.iteration)
                if not self.use_streamdata:
                    msg += "batch: {}/{}, ".format(i + 1,
                                                   len(self.valid_loader))
                msg += ', '.join('{}: {:>.6f}'.format(k, v)
                                 for k, v in valid_losses.items())
                logger.info(msg)

        logger.info(
            'Rank {} Val info val_loss {}'.format(dist.get_rank(), total_loss))
        return total_loss, num_seen_utts

    @mp_tools.rank_zero_only
    def save(self, tag=None, infos: dict=None):
        """Save checkpoint (model parameters and optimizer states).

        Args:
            tag (int or str, optional): None for step, else using tag, e.g epoch. Defaults to None.
            infos (dict, optional): meta data to save. Defaults to None.
        """

        infos = infos if infos else dict()
        infos.update({
            "epoch": self.epoch,
            "model_lr": self.model_optimizer.get_lr(),
            "wavlm_lr": self.wavlm_optimizer.get_lr()
        })

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            "{}".format(self.iteration if tag is None else tag))

        model_dict = self.model.state_dict()
        params_path = checkpoint_path + ".pdparams"
        paddle.save(model_dict, params_path)
        logger.info("Saved model to {}".format(params_path))

        model_opt_dict = self.model_optimizer.state_dict()
        wavlm_opt_dict = self.wavlm_optimizer.state_dict()

        opt_dict = {'model': model_opt_dict, 'wavlm': wavlm_opt_dict}

        optimizer_path = checkpoint_path + ".pdopt"
        paddle.save(opt_dict, optimizer_path)
        logger.info("Saved optimzier state to {}".format(optimizer_path))

        scheduler_dict = {}

        if self.config.model_scheduler == 'newbobscheduler':
            scheduler_dict['model'] = self.model_lr_scheduler.save()
        if self.config.wavlm_scheduler == 'newbobscheduler':
            scheduler_dict['wavlm'] = self.wavlm_lr_scheduler.save()
        if scheduler_dict:
            scheduler_path = checkpoint_path + ".pdlrs"
            paddle.save(scheduler_dict, scheduler_path)
            logger.info("Saved scheduler state to {}".format(scheduler_path))
        info_path = re.sub('.pdparams$', '.json', params_path)
        infos = {} if infos is None else infos
        with open(info_path, 'w', encoding='utf8') as fout:
            data = json.dumps(infos)
            fout.write(data)

    def resume_or_scratch(self):
        """Resume from latest checkpoint at checkpoints in the output
        directory or load a specified checkpoint.

        If ``args.checkpoint_path`` is not None, load the checkpoint, else
        resume training.
        """
        scratch = None
        if self.args.resume:
            # just restore ckpt
            # lr will resotre from optimizer ckpt
            resume_json_path = os.path.join(self.checkpoint_dir,
                                            self.args.resume + '.json')
            with open(resume_json_path, 'r', encoding='utf8') as f:
                resume_json = json.load(f)
            self.iteration = 0
            self.epoch = resume_json["epoch"]

            # resotre model from *.pdparams
            params_path = os.path.join(self.checkpoint_dir,
                                       "{}".format(self.epoch)) + '.pdparams'
            model_dict = paddle.load(params_path)
            self.model.set_state_dict(model_dict)

            # resotre optimizer from *.pdopt
            optimizer_path = os.path.join(self.checkpoint_dir,
                                          "{}".format(self.epoch)) + '.pdopt'
            optimizer_dict = paddle.load(optimizer_path)
            self.model_optimizer.set_state_dict(optimizer_dict['model'])
            self.wavlm_optimizer.set_state_dict(optimizer_dict['wavlm'])

            # resotre lr_scheduler from *.pdlrs
            scheduler_path = os.path.join(self.checkpoint_dir,
                                          "{}".format(self.epoch)) + '.pdlrs'
            if os.path.isfile(os.path.join(scheduler_path)):
                scheduler_dict = paddle.load(scheduler_path)
                if self.config.model_scheduler == 'newbobscheduler':
                    self.model_lr_scheduler.load(scheduler_dict['model'])
                if self.config.wavlm_scheduler == 'newbobscheduler':
                    self.wavlm_lr_scheduler.load(scheduler_dict['wavlm'])
            logger.info(
                f"Restore ckpt: epoch {self.epoch }, step {self.iteration}!")
            scratch = False
        else:
            self.iteration = 0
            self.epoch = 0
            scratch = True
            logger.info("Init from scratch!")
        return scratch

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
                            report("model_lr", self.model_optimizer.get_lr())
                            report("wavlm_lr", self.wavlm_optimizer.get_lr())
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
                    dist.all_reduce(num_seen_utts)
                    total_loss = paddle.to_tensor(total_loss)
                    dist.all_reduce(total_loss)
                    cv_loss = total_loss / num_seen_utts
                    cv_loss = float(cv_loss)
                else:
                    cv_loss = float(total_loss)
            logger.info(
                'Epoch {} Val info val_loss {}'.format(self.epoch, cv_loss))
            if self.visualizer:
                self.visualizer.add_scalar(
                    tag='eval/cv_loss', value=cv_loss, step=self.epoch)
                self.visualizer.add_scalar(
                    tag='eval/model_lr',
                    value=self.model_lr_scheduler(),
                    step=self.epoch)
                self.visualizer.add_scalar(
                    tag='eval/wavlm_lr',
                    value=self.wavlm_lr_scheduler(),
                    step=self.epoch)

            if self.config.model_scheduler == 'newbobscheduler':
                self.model_lr_scheduler.step(cv_loss)
            if self.config.wavlm_scheduler == 'newbobscheduler':
                if not self.config.freeze_wavlm:
                    self.wavlm_lr_scheduler.step(cv_loss)
            self.save(tag=self.epoch, infos={'val_loss': cv_loss})
            self.avg_train_loss = 0.0
            self.new_epoch()

    def dataio_prepare(self, hparams):
        """This function prepares the datasets to be used in the brain class.
        It also defines the data processing pipeline through user-defined functions."""
        data_folder = hparams["data_folder"]

        train_data = dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["train_data"],
            replacements={"data_root": data_folder}, )

        if hparams["sorting"] == "ascending":
            # we sort training data to speed up training and get better results.
            train_data = train_data.filtered_sorted(sort_key="duration")
            # when sorting do not shuffle in dataloader ! otherwise is pointless
            hparams["train_dataloader_opts"]["shuffle"] = False

        elif hparams["sorting"] == "descending":
            train_data = train_data.filtered_sorted(
                sort_key="duration", reverse=True)
            # when sorting do not shuffle in dataloader ! otherwise is pointless
            hparams["train_dataloader_opts"]["shuffle"] = False

        elif hparams["sorting"] == "random":
            pass

        else:
            raise NotImplementedError(
                "sorting must be random, ascending or descending")

        valid_data = dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["valid_data"],
            replacements={"data_root": data_folder}, )
        valid_data = valid_data.filtered_sorted(sort_key="duration")

        test_data = dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["test_data"],
            replacements={"data_root": data_folder}, )
        test_data = test_data.filtered_sorted(sort_key="duration")

        datasets = [train_data, valid_data, test_data]

        # Defining tokenizer and loading it
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.tokenizer = tokenizer
        # 2. Define audio pipeline:
        @data_pipeline.takes("wav")
        @data_pipeline.provides("sig")
        def audio_pipeline(wav):
            sig = dataio.read_audio(wav)
            return sig

        dataset.add_dynamic_item(datasets, audio_pipeline)

        # 3. Define text pipeline:
        @data_pipeline.takes("transcript")
        @data_pipeline.provides("wrd", "tokens_list", "tokens")
        def text_pipeline(wrd):
            wrd = "".join(wrd.split(" "))
            yield wrd
            tokens_list = tokenizer(wrd)["input_ids"]
            yield tokens_list
            tokens = np.array(tokens_list, dtype="int64")
            # tokens = paddle.to_tensor(tokens_list, dtype="int64")
            yield tokens

        dataset.add_dynamic_item(datasets, text_pipeline)

        # 4. Set output:
        dataset.set_output_keys(
            datasets,
            ["id", "sig", "wrd", "tokens"], )

        # 5. If Dynamic Batching is used, we instantiate the needed samplers.
        train_batch_sampler = None
        valid_batch_sampler = None
        if hparams["dynamic_batching"]:
            from sampler import DynamicBatchSampler  # noqa

            dynamic_hparams = hparams["dynamic_batch_sampler"]
            num_buckets = dynamic_hparams["num_buckets"]

            train_batch_sampler = DynamicBatchSampler(
                train_data,
                dynamic_hparams["max_batch_len"],
                num_buckets=num_buckets,
                length_func=lambda x: x["duration"],
                shuffle=dynamic_hparams["shuffle_ex"],
                batch_ordering=dynamic_hparams["batch_ordering"], )

            valid_batch_sampler = DynamicBatchSampler(
                valid_data,
                dynamic_hparams["max_batch_len"],
                num_buckets=num_buckets,
                length_func=lambda x: x["duration"],
                shuffle=dynamic_hparams["shuffle_ex"],
                batch_ordering=dynamic_hparams["batch_ordering"], )

        return (train_data, valid_data, test_data, tokenizer,
                train_batch_sampler, valid_batch_sampler, )

    def setup_dataloader(self):
        config = self.config.clone()
        self.use_streamdata = config.get("use_stream_data", False)
        self.use_sb = config.get("use_sb_pipeline", False)
        if self.use_sb:
            hparams_file = config.sb_pipeline_conf
            with open(hparams_file, 'r', encoding='utf8') as fin:
                hparams = load_hyperpyyaml(fin, None)

            (train_data, valid_data, test_data, tokenizer, train_bsampler,
             valid_bsampler, ) = self.dataio_prepare(hparams)

            train_dataloader_opts = hparams["train_dataloader_opts"]
            valid_dataloader_opts = hparams["valid_dataloader_opts"]

            if train_bsampler is not None:
                train_dataloader_opts = {
                    "batch_sampler": train_bsampler,
                    "num_workers": hparams["num_workers"],
                }

            if valid_bsampler is not None:
                valid_dataloader_opts = {"batch_sampler": valid_bsampler}

            if self.train:
                self.train_loader = make_dataloader(
                    train_data, stage='train', **train_dataloader_opts)
                self.valid_loader = make_dataloader(
                    valid_data,
                    stage='val',
                    **valid_dataloader_opts, )
                logger.info("Setup train/valid Dataloader!")
            else:
                self.test_loader = make_dataloader(
                    test_data, stage='test', **hparams["test_dataloader_opts"])
        else:
            if self.train:
                self.train_loader = DataLoaderFactory.get_dataloader(
                    'train', config, self.args)
                self.valid_loader = DataLoaderFactory.get_dataloader(
                    'valid', config, self.args)
                logger.info("Setup train/valid Dataloader!")
            else:
                decode_batch_size = config.get('decode', dict()).get(
                    'decode_batch_size', 1)
                self.test_loader = DataLoaderFactory.get_dataloader(
                    'test', config, self.args)
                self.align_loader = DataLoaderFactory.get_dataloader(
                    'align', config, self.args)
                logger.info("Setup test/align Dataloader!")

    def setup_model(self):
        config = self.config
        model_conf = config

        with UpdateConfig(model_conf):
            if self.use_sb:
                model_conf.output_dim = self.tokenizer.vocab_size
            else:
                if self.train:
                    model_conf.input_dim = self.train_loader.feat_dim
                    model_conf.output_dim = self.train_loader.vocab_size
                else:
                    model_conf.input_dim = self.test_loader.feat_dim
                    model_conf.output_dim = self.test_loader.vocab_size

        model = WavLMASR.from_config(model_conf)

        model_dict = paddle.load(config.wavlm_params_path)
        model.wavlm.set_state_dict(model_dict)

        if self.parallel:
            model = paddle.DataParallel(model, find_unused_parameters=True)

        layer_tools.print_params(model, logger.info)
        self.model = model
        logger.info("Setup model!")

        # setup speech augmentation for wavlm
        if hasattr(config, 'audio_augment') and self.train:
            self.speech_augmentation = TimeDomainSpecAugment(
                **config.audio_augment)

        if not self.train:
            return

        train_config = config
        model_optim_type = train_config.model_optim
        model_optim_conf = train_config.model_optim_conf
        logger.info("optim_model:{},{}", model_optim_type, model_optim_conf)
        wavlm_optim_type = train_config.wavlm_optim
        wavlm_optim_conf = train_config.wavlm_optim_conf
        logger.info("optim_model:{},{}", wavlm_optim_type, wavlm_optim_conf)

        model_scheduler_type = train_config.model_scheduler
        model_scheduler_conf = train_config.model_scheduler_conf
        wavlm_scheduler_type = train_config.wavlm_scheduler
        wavlm_scheduler_conf = train_config.wavlm_scheduler_conf

        model_scheduler_args = dict(
            **{"learning_rate": model_optim_conf.lr,
               "verbose": False}, **(dict(model_scheduler_conf)))

        wavlm_scheduler_args = dict(
            **{"learning_rate": wavlm_optim_conf.lr,
               "verbose": False}, **(dict(wavlm_scheduler_conf)))

        model_lr_scheduler = LRSchedulerFactory.from_args(model_scheduler_type,
                                                          model_scheduler_args)
        wavlm_lr_scheduler = LRSchedulerFactory.from_args(wavlm_scheduler_type,
                                                          wavlm_scheduler_args)

        def optimizer_args(
                config,
                optim_type,
                optim_conf,
                parameters,
                lr_scheduler=None, ):
            optim_arg = dict(optim_conf)
            optim_arg.update({
                "learning_rate":
                lr_scheduler if lr_scheduler else optim_conf.lr,
                "parameters":
                parameters
            })
            return optim_arg

        model_optimizer_args = optimizer_args(config, model_optim_type,
                                              model_optim_conf, [{
                                                  'params':
                                                  model._layers.enc.parameters()
                                              }, {
                                                  'params':
                                                  model._layers.ctc.parameters()
                                              }] if self.parallel else [{
                                                  'params':
                                                  model.enc.parameters()
                                              }, {
                                                  'params':
                                                  model.ctc.parameters()
                                              }], model_lr_scheduler)
        # [{'params': model._layers.ctc.parameters()}] if self.parallel else [{'params': model.ctc.parameters()}], model_lr_scheduler)

        wavlm_optimizer_args = optimizer_args(
            config, wavlm_optim_type, wavlm_optim_conf,
            model._layers.wavlm.parameters()
            if self.parallel else model.wavlm.parameters(), wavlm_lr_scheduler)

        model_optimizer = OptimizerFactory.from_args(model_optim_type,
                                                     model_optimizer_args)
        wavlm_optimizer = OptimizerFactory.from_args(wavlm_optim_type,
                                                     wavlm_optimizer_args)

        self.model_optimizer = model_optimizer
        self.wavlm_optimizer = wavlm_optimizer
        self.model_lr_scheduler = model_lr_scheduler
        self.wavlm_lr_scheduler = wavlm_lr_scheduler
        logger.info("Setup optimizer/lr_scheduler!")


class WavLMASRTester(WavLMASRTrainer):
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

    def compute_metrics(self, id, audio, audio_len, texts, texts_len,
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
                id, target_transcripts, result_transcripts, result_tokenids):
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

    def sb_compute_metrics(self, id, sig, wrd, tokens, fout=None):
        decode_cfg = self.config.decode
        errors_sum, len_refs, num_ins = 0.0, 0, 0
        errors_func = error_rate.char_errors if decode_cfg.error_rate_type == 'cer' else error_rate.word_errors
        error_rate_func = error_rate.cer if decode_cfg.error_rate_type == 'cer' else error_rate.wer
        start_time = time.time()
        target_transcripts = wrd
        result_transcripts, result_tokenids = self.model.decode(
            sig[0],
            text_feature=self.tokenizer,
            decoding_method=decode_cfg.decoding_method,
            beam_size=decode_cfg.beam_size,
            sb_pipeline=True)
        decode_time = time.time() - start_time

        for utt, target, result, rec_tids in zip(
                id, target_transcripts, result_transcripts, result_tokenids):
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
            num_frames=sig[1].sum().numpy().item(),
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
                if self.use_sb:
                    metrics = self.sb_compute_metrics(**batch, fout=fout)
                else:
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
        with open(err_meta_path, 'w', encoding='utf8') as f:
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
