# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import logging
from typing import Dict

import paddle
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.nn import Layer
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler

from paddlespeech.t2s.modules.nets_utils import get_segments
from paddlespeech.t2s.training.extensions.evaluator import StandardEvaluator
from paddlespeech.t2s.training.reporter import report
from paddlespeech.t2s.training.updaters.standard_updater import StandardUpdater
from paddlespeech.t2s.training.updaters.standard_updater import UpdaterState

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VITSUpdater(StandardUpdater):
    def __init__(self,
                 model: Layer,
                 optimizers: Dict[str, Optimizer],
                 criterions: Dict[str, Layer],
                 schedulers: Dict[str, LRScheduler],
                 dataloader: DataLoader,
                 generator_train_start_steps: int=0,
                 discriminator_train_start_steps: int=100000,
                 lambda_adv: float=1.0,
                 lambda_mel: float=45.0,
                 lambda_feat_match: float=2.0,
                 lambda_dur: float=1.0,
                 lambda_kl: float=1.0,
                 generator_first: bool=False,
                 output_dir=None):
        # it is designed to hold multiple models
        # 因为输入的是单模型，但是没有用到父类的 init(), 所以需要重新写这部分
        models = {"main": model}
        self.models: Dict[str, Layer] = models
        # self.model = model

        self.model = model._layers if isinstance(model, paddle.DataParallel) else model

        self.optimizers = optimizers
        self.optimizer_g: Optimizer = optimizers['generator']
        self.optimizer_d: Optimizer = optimizers['discriminator']

        self.criterions = criterions
        self.criterion_mel = criterions['mel']
        self.criterion_feat_match = criterions['feat_match']
        self.criterion_gen_adv = criterions["gen_adv"]
        self.criterion_dis_adv = criterions["dis_adv"]
        self.criterion_kl = criterions["kl"]

        self.schedulers = schedulers
        self.scheduler_g = schedulers['generator']
        self.scheduler_d = schedulers['discriminator']

        self.dataloader = dataloader

        self.generator_train_start_steps = generator_train_start_steps
        self.discriminator_train_start_steps = discriminator_train_start_steps

        self.lambda_adv = lambda_adv
        self.lambda_mel = lambda_mel
        self.lambda_feat_match = lambda_feat_match
        self.lambda_dur = lambda_dur
        self.lambda_kl = lambda_kl

        if generator_first:
            self.turns = ["generator", "discriminator"]
        else:
            self.turns = ["discriminator", "generator"]

        self.state = UpdaterState(iteration=0, epoch=0)
        self.train_iterator = iter(self.dataloader)

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def update_core(self, batch):
        self.msg = "Rank: {}, ".format(dist.get_rank())
        losses_dict = {}

        for turn in self.turns:
            speech = batch["speech"]
            speech = speech.unsqueeze(1)
            outs = self.model(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                feats=batch["feats"],
                feats_lengths=batch["feats_lengths"],
                forward_generator=turn == "generator")
            # Generator
            if turn == "generator":
                # parse outputs
                speech_hat_, dur_nll, _, start_idxs, _, z_mask, outs_ = outs
                _, z_p, m_p, logs_p, _, logs_q = outs_
                speech_ = get_segments(
                    x=speech,
                    start_idxs=start_idxs *
                    self.model.generator.upsample_factor,
                    segment_size=self.model.generator.segment_size *
                    self.model.generator.upsample_factor, )

                # calculate discriminator outputs
                p_hat = self.model.discriminator(speech_hat_)
                with paddle.no_grad():
                    # do not store discriminator gradient in generator turn
                    p = self.model.discriminator(speech_)

                # calculate losses
                mel_loss = self.criterion_mel(speech_hat_, speech_)
                kl_loss = self.criterion_kl(z_p, logs_q, m_p, logs_p, z_mask)
                dur_loss = paddle.sum(dur_nll)
                adv_loss = self.criterion_gen_adv(p_hat)
                feat_match_loss = self.criterion_feat_match(p_hat, p)

                mel_loss = mel_loss * self.lambda_mel
                kl_loss = kl_loss * self.lambda_kl
                dur_loss = dur_loss * self.lambda_dur
                adv_loss = adv_loss * self.lambda_adv
                feat_match_loss = feat_match_loss * self.lambda_feat_match
                gen_loss = mel_loss + kl_loss + dur_loss + adv_loss + feat_match_loss

                report("train/generator_loss", float(gen_loss))
                report("train/generator_mel_loss", float(mel_loss))
                report("train/generator_kl_loss", float(kl_loss))
                report("train/generator_dur_loss", float(dur_loss))
                report("train/generator_adv_loss", float(adv_loss))
                report("train/generator_feat_match_loss",
                       float(feat_match_loss))

                losses_dict["generator_loss"] = float(gen_loss)
                losses_dict["generator_mel_loss"] = float(mel_loss)
                losses_dict["generator_kl_loss"] = float(kl_loss)
                losses_dict["generator_dur_loss"] = float(dur_loss)
                losses_dict["generator_adv_loss"] = float(adv_loss)
                losses_dict["generator_feat_match_loss"] = float(
                    feat_match_loss)

                self.optimizer_g.clear_grad()
                gen_loss.backward()

                self.optimizer_g.step()
                self.scheduler_g.step()

                # reset cache
                if self.model.reuse_cache_gen or not self.model.training:
                    self.model._cache = None

            # Disctiminator
            elif turn == "discriminator":
                # parse outputs
                speech_hat_, _, _, start_idxs, *_ = outs
                speech_ = get_segments(
                    x=speech,
                    start_idxs=start_idxs *
                    self.model.generator.upsample_factor,
                    segment_size=self.model.generator.segment_size *
                    self.model.generator.upsample_factor, )

                # calculate discriminator outputs
                p_hat = self.model.discriminator(speech_hat_.detach())
                p = self.model.discriminator(speech_)

                # calculate losses
                real_loss, fake_loss = self.criterion_dis_adv(p_hat, p)
                dis_loss = real_loss + fake_loss

                report("train/real_loss", float(real_loss))
                report("train/fake_loss", float(fake_loss))
                report("train/discriminator_loss", float(dis_loss))
                losses_dict["real_loss"] = float(real_loss)
                losses_dict["fake_loss"] = float(fake_loss)
                losses_dict["discriminator_loss"] = float(dis_loss)

                self.optimizer_d.clear_grad()
                dis_loss.backward()

                self.optimizer_d.step()
                self.scheduler_d.step()

                # reset cache
                if self.model.reuse_cache_dis or not self.model.training:
                    self.model._cache = None

        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())


class VITSEvaluator(StandardEvaluator):
    def __init__(self,
                 model,
                 criterions: Dict[str, Layer],
                 dataloader: DataLoader,
                 lambda_adv: float=1.0,
                 lambda_mel: float=45.0,
                 lambda_feat_match: float=2.0,
                 lambda_dur: float=1.0,
                 lambda_kl: float=1.0,
                 generator_first: bool=False,
                 output_dir=None):
        # 因为输入的是单模型，但是没有用到父类的 init(), 所以需要重新写这部分
        models = {"main": model}
        self.models: Dict[str, Layer] = models
        # self.model = model
        self.model = model._layers if isinstance(model, paddle.DataParallel) else model

        self.criterions = criterions
        self.criterion_mel = criterions['mel']
        self.criterion_feat_match = criterions['feat_match']
        self.criterion_gen_adv = criterions["gen_adv"]
        self.criterion_dis_adv = criterions["dis_adv"]
        self.criterion_kl = criterions["kl"]

        self.dataloader = dataloader

        self.lambda_adv = lambda_adv
        self.lambda_mel = lambda_mel
        self.lambda_feat_match = lambda_feat_match
        self.lambda_dur = lambda_dur
        self.lambda_kl = lambda_kl

        if generator_first:
            self.turns = ["generator", "discriminator"]
        else:
            self.turns = ["discriminator", "generator"]

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def evaluate_core(self, batch):
        # logging.debug("Evaluate: ")
        self.msg = "Evaluate: "
        losses_dict = {}

        for turn in self.turns:
            speech = batch["speech"]
            speech = speech.unsqueeze(1)
            outs = self.model(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                feats=batch["feats"],
                feats_lengths=batch["feats_lengths"],
                forward_generator=turn == "generator")
            # Generator
            if turn == "generator":
                # parse outputs
                speech_hat_, dur_nll, _, start_idxs, _, z_mask, outs_ = outs
                _, z_p, m_p, logs_p, _, logs_q = outs_
                speech_ = get_segments(
                    x=speech,
                    start_idxs=start_idxs *
                    self.model.generator.upsample_factor,
                    segment_size=self.model.generator.segment_size *
                    self.model.generator.upsample_factor, )

                # calculate discriminator outputs
                p_hat = self.model.discriminator(speech_hat_)
                with paddle.no_grad():
                    # do not store discriminator gradient in generator turn
                    p = self.model.discriminator(speech_)

                # calculate losses
                mel_loss = self.criterion_mel(speech_hat_, speech_)
                kl_loss = self.criterion_kl(z_p, logs_q, m_p, logs_p, z_mask)
                dur_loss = paddle.sum(dur_nll)
                adv_loss = self.criterion_gen_adv(p_hat)
                feat_match_loss = self.criterion_feat_match(p_hat, p)

                mel_loss = mel_loss * self.lambda_mel
                kl_loss = kl_loss * self.lambda_kl
                dur_loss = dur_loss * self.lambda_dur
                adv_loss = adv_loss * self.lambda_adv
                feat_match_loss = feat_match_loss * self.lambda_feat_match
                gen_loss = mel_loss + kl_loss + dur_loss + adv_loss + feat_match_loss

                report("eval/generator_loss", float(gen_loss))
                report("eval/generator_mel_loss", float(mel_loss))
                report("eval/generator_kl_loss", float(kl_loss))
                report("eval/generator_dur_loss", float(dur_loss))
                report("eval/generator_adv_loss", float(adv_loss))
                report("eval/generator_feat_match_loss", float(feat_match_loss))

                losses_dict["generator_loss"] = float(gen_loss)
                losses_dict["generator_mel_loss"] = float(mel_loss)
                losses_dict["generator_kl_loss"] = float(kl_loss)
                losses_dict["generator_dur_loss"] = float(dur_loss)
                losses_dict["generator_adv_loss"] = float(adv_loss)
                losses_dict["generator_feat_match_loss"] = float(
                    feat_match_loss)

                # reset cache
                if self.model.reuse_cache_gen or not self.model.training:
                    self.model._cache = None

            # Disctiminator
            elif turn == "discriminator":
                # parse outputs
                speech_hat_, _, _, start_idxs, *_ = outs
                speech_ = get_segments(
                    x=speech,
                    start_idxs=start_idxs *
                    self.model.generator.upsample_factor,
                    segment_size=self.model.generator.segment_size *
                    self.model.generator.upsample_factor, )

                # calculate discriminator outputs
                p_hat = self.model.discriminator(speech_hat_.detach())
                p = self.model.discriminator(speech_)

                # calculate losses
                real_loss, fake_loss = self.criterion_dis_adv(p_hat, p)
                dis_loss = real_loss + fake_loss

                report("eval/real_loss", float(real_loss))
                report("eval/fake_loss", float(fake_loss))
                report("eval/discriminator_loss", float(dis_loss))
                losses_dict["real_loss"] = float(real_loss)
                losses_dict["fake_loss"] = float(fake_loss)
                losses_dict["discriminator_loss"] = float(dis_loss)

                # reset cache
                if self.model.reuse_cache_dis or not self.model.training:
                    self.model._cache = None

        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        self.logger.info(self.msg)
