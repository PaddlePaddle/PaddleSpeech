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
"""Generator module in JETS.

This code is based on https://github.com/imdanboy/jets.

"""
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


class JETSUpdater(StandardUpdater):
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
                 lambda_var: float=1.0,
                 lambda_align: float=2.0,
                 generator_first: bool=False,
                 use_alignment_module: bool=False,
                 output_dir=None):
        # it is designed to hold multiple models
        # 因为输入的是单模型，但是没有用到父类的 init(), 所以需要重新写这部分
        models = {"main": model}
        self.models: Dict[str, Layer] = models
        # self.model = model

        self.model = model._layers if isinstance(model,
                                                 paddle.DataParallel) else model

        self.optimizers = optimizers
        self.optimizer_g: Optimizer = optimizers['generator']
        self.optimizer_d: Optimizer = optimizers['discriminator']

        self.criterions = criterions
        self.criterion_mel = criterions['mel']
        self.criterion_feat_match = criterions['feat_match']
        self.criterion_gen_adv = criterions["gen_adv"]
        self.criterion_dis_adv = criterions["dis_adv"]
        self.criterion_var = criterions["var"]
        self.criterion_forwardsum = criterions["forwardsum"]

        self.schedulers = schedulers
        self.scheduler_g = schedulers['generator']
        self.scheduler_d = schedulers['discriminator']

        self.dataloader = dataloader

        self.generator_train_start_steps = generator_train_start_steps
        self.discriminator_train_start_steps = discriminator_train_start_steps

        self.lambda_adv = lambda_adv
        self.lambda_mel = lambda_mel
        self.lambda_feat_match = lambda_feat_match
        self.lambda_var = lambda_var
        self.lambda_align = lambda_align

        self.use_alignment_module = use_alignment_module

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
            text_lengths = batch["text_lengths"]
            feats_lengths = batch["feats_lengths"]
            outs = self.model(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                feats=batch["feats"],
                feats_lengths=batch["feats_lengths"],
                durations=batch["durations"],
                durations_lengths=batch["durations_lengths"],
                pitch=batch["pitch"],
                energy=batch["energy"],
                sids=batch.get("spk_id", None),
                spembs=batch.get("spk_emb", None),
                forward_generator=turn == "generator",
                use_alignment_module=self.use_alignment_module)
            # Generator
            if turn == "generator":
                # parse outputs
                speech_hat_, bin_loss, log_p_attn, start_idxs, d_outs, ds, p_outs, ps, e_outs, es = outs
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

                adv_loss = self.criterion_gen_adv(p_hat)
                feat_match_loss = self.criterion_feat_match(p_hat, p)
                dur_loss, pitch_loss, energy_loss = self.criterion_var(
                    d_outs, ds, p_outs, ps, e_outs, es, text_lengths)

                mel_loss = mel_loss * self.lambda_mel
                adv_loss = adv_loss * self.lambda_adv
                feat_match_loss = feat_match_loss * self.lambda_feat_match
                g_loss = mel_loss + adv_loss + feat_match_loss
                var_loss = (
                    dur_loss + pitch_loss + energy_loss) * self.lambda_var

                gen_loss = g_loss + var_loss  #+ align_loss

                report("train/generator_loss", float(gen_loss))
                report("train/generator_generator_loss", float(g_loss))
                report("train/generator_variance_loss", float(var_loss))
                report("train/generator_generator_mel_loss", float(mel_loss))
                report("train/generator_generator_adv_loss", float(adv_loss))
                report("train/generator_generator_feat_match_loss",
                       float(feat_match_loss))
                report("train/generator_variance_dur_loss", float(dur_loss))
                report("train/generator_variance_pitch_loss", float(pitch_loss))
                report("train/generator_variance_energy_loss",
                       float(energy_loss))

                losses_dict["generator_loss"] = float(gen_loss)
                losses_dict["generator_generator_loss"] = float(g_loss)
                losses_dict["generator_variance_loss"] = float(var_loss)
                losses_dict["generator_generator_mel_loss"] = float(mel_loss)
                losses_dict["generator_generator_adv_loss"] = float(adv_loss)
                losses_dict["generator_generator_feat_match_loss"] = float(
                    feat_match_loss)
                losses_dict["generator_variance_dur_loss"] = float(dur_loss)
                losses_dict["generator_variance_pitch_loss"] = float(pitch_loss)
                losses_dict["generator_variance_energy_loss"] = float(
                    energy_loss)

                if self.use_alignment_module == True:
                    forwardsum_loss = self.criterion_forwardsum(
                        log_p_attn, text_lengths, feats_lengths)
                    align_loss = (
                        forwardsum_loss + bin_loss) * self.lambda_align
                    report("train/generator_alignment_loss", float(align_loss))
                    report("train/generator_alignment_forwardsum_loss",
                           float(forwardsum_loss))
                    report("train/generator_alignment_bin_loss",
                           float(bin_loss))
                    losses_dict["generator_alignment_loss"] = float(align_loss)
                    losses_dict["generator_alignment_forwardsum_loss"] = float(
                        forwardsum_loss)
                    losses_dict["generator_alignment_bin_loss"] = float(
                        bin_loss)

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


class JETSEvaluator(StandardEvaluator):
    def __init__(self,
                 model,
                 criterions: Dict[str, Layer],
                 dataloader: DataLoader,
                 lambda_adv: float=1.0,
                 lambda_mel: float=45.0,
                 lambda_feat_match: float=2.0,
                 lambda_var: float=1.0,
                 lambda_align: float=2.0,
                 generator_first: bool=False,
                 use_alignment_module: bool=False,
                 output_dir=None):
        # 因为输入的是单模型，但是没有用到父类的 init(), 所以需要重新写这部分
        models = {"main": model}
        self.models: Dict[str, Layer] = models
        # self.model = model
        self.model = model._layers if isinstance(model,
                                                 paddle.DataParallel) else model

        self.criterions = criterions
        self.criterion_mel = criterions['mel']
        self.criterion_feat_match = criterions['feat_match']
        self.criterion_gen_adv = criterions["gen_adv"]
        self.criterion_dis_adv = criterions["dis_adv"]
        self.criterion_var = criterions["var"]
        self.criterion_forwardsum = criterions["forwardsum"]

        self.dataloader = dataloader

        self.lambda_adv = lambda_adv
        self.lambda_mel = lambda_mel
        self.lambda_feat_match = lambda_feat_match
        self.lambda_var = lambda_var
        self.lambda_align = lambda_align
        self.use_alignment_module = use_alignment_module

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
            text_lengths = batch["text_lengths"]
            feats_lengths = batch["feats_lengths"]
            outs = self.model(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                feats=batch["feats"],
                feats_lengths=batch["feats_lengths"],
                durations=batch["durations"],
                durations_lengths=batch["durations_lengths"],
                pitch=batch["pitch"],
                energy=batch["energy"],
                sids=batch.get("spk_id", None),
                spembs=batch.get("spk_emb", None),
                forward_generator=turn == "generator",
                use_alignment_module=self.use_alignment_module)
            # Generator
            if turn == "generator":
                # parse outputs
                speech_hat_, bin_loss, log_p_attn, start_idxs, d_outs, ds, p_outs, ps, e_outs, es = outs
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

                adv_loss = self.criterion_gen_adv(p_hat)
                feat_match_loss = self.criterion_feat_match(p_hat, p)
                dur_loss, pitch_loss, energy_loss = self.criterion_var(
                    d_outs, ds, p_outs, ps, e_outs, es, text_lengths)

                mel_loss = mel_loss * self.lambda_mel
                adv_loss = adv_loss * self.lambda_adv
                feat_match_loss = feat_match_loss * self.lambda_feat_match
                g_loss = mel_loss + adv_loss + feat_match_loss
                var_loss = (
                    dur_loss + pitch_loss + energy_loss) * self.lambda_var

                gen_loss = g_loss + var_loss  #+ align_loss

                report("eval/generator_loss", float(gen_loss))
                report("eval/generator_generator_loss", float(g_loss))
                report("eval/generator_variance_loss", float(var_loss))
                report("eval/generator_generator_mel_loss", float(mel_loss))
                report("eval/generator_generator_adv_loss", float(adv_loss))
                report("eval/generator_generator_feat_match_loss",
                       float(feat_match_loss))
                report("eval/generator_variance_dur_loss", float(dur_loss))
                report("eval/generator_variance_pitch_loss", float(pitch_loss))
                report("eval/generator_variance_energy_loss",
                       float(energy_loss))

                losses_dict["generator_loss"] = float(gen_loss)
                losses_dict["generator_generator_loss"] = float(g_loss)
                losses_dict["generator_variance_loss"] = float(var_loss)
                losses_dict["generator_generator_mel_loss"] = float(mel_loss)
                losses_dict["generator_generator_adv_loss"] = float(adv_loss)
                losses_dict["generator_generator_feat_match_loss"] = float(
                    feat_match_loss)
                losses_dict["generator_variance_dur_loss"] = float(dur_loss)
                losses_dict["generator_variance_pitch_loss"] = float(pitch_loss)
                losses_dict["generator_variance_energy_loss"] = float(
                    energy_loss)

                if self.use_alignment_module == True:
                    forwardsum_loss = self.criterion_forwardsum(
                        log_p_attn, text_lengths, feats_lengths)
                    align_loss = (
                        forwardsum_loss + bin_loss) * self.lambda_align
                    report("eval/generator_alignment_loss", float(align_loss))
                    report("eval/generator_alignment_forwardsum_loss",
                           float(forwardsum_loss))
                    report("eval/generator_alignment_bin_loss", float(bin_loss))
                    losses_dict["generator_alignment_loss"] = float(align_loss)
                    losses_dict["generator_alignment_forwardsum_loss"] = float(
                        forwardsum_loss)
                    losses_dict["generator_alignment_bin_loss"] = float(
                        bin_loss)

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
