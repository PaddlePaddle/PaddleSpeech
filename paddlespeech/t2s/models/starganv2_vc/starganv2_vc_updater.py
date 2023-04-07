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
import logging
from typing import Any
from typing import Dict

import paddle
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.nn import Layer
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler

from paddlespeech.t2s.training.extensions.evaluator import StandardEvaluator
from paddlespeech.t2s.training.reporter import report
from paddlespeech.t2s.training.updaters.standard_updater import StandardUpdater
from paddlespeech.t2s.training.updaters.standard_updater import UpdaterState
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StarGANv2VCUpdater(StandardUpdater):
    def __init__(self,
                 models: Dict[str, Layer],
                 optimizers: Dict[str, Optimizer],
                 criterions: Dict[str, Layer],
                 schedulers: Dict[str, LRScheduler],
                 dataloader: DataLoader,
                 g_loss_params: Dict[str, Any]={
                     'lambda_sty': 1.,
                     'lambda_cyc': 5.,
                     'lambda_ds': 1.,
                     'lambda_norm': 1.,
                     'lambda_asr': 10.,
                     'lambda_f0': 5.,
                     'lambda_f0_sty': 0.1,
                     'lambda_adv': 2.,
                     'lambda_adv_cls': 0.5,
                     'norm_bias': 0.5,
                 },
                 d_loss_params: Dict[str, Any]={
                     'lambda_reg': 1.,
                     'lambda_adv_cls': 0.1,
                     'lambda_con_reg': 10.,
                 },
                 output_dir=None):
        self.models = models
        self.generator = models['generator']
        self.style_encoder = models['style_encoder']
        self.mapping_network = models['mapping_network']
        self.discriminator = models['discriminator']
        self.F0_model = models['F0_model']
        self.asr_model = models['asr_model']

        self.optimizers = optimizers
        self.optimizer_g = optimizers['optimizer_g']
        self.optimizer_s = optimizers['optimizer_s']
        self.optimizer_m = optimizers['optimizer_m']
        self.optimizer_d = optimizers['optimizer_d']

        self.criterions = criterions
        self.criterion_f0 = criterions['f0']
        self.criterion_r1_reg = criterions['r1_reg']
        self.criterion_adv = criterions["adv"]

        self.schedulers = schedulers
        self.scheduler_g = schedulers['generator']
        self.scheduler_s = schedulers['style_encoder']
        self.scheduler_m = schedulers['mapping_network']
        self.scheduler_d = schedulers['discriminator']

        self.dataloader = dataloader

        self.g_loss_params = g_loss_params
        self.d_loss_params = d_loss_params

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
        # parse batch
        x_real, y_org, x_ref, x_ref2, y_trg, z_trg, z_trg2 = batch

        # Discriminator

        # Generator
        '''
        if self.state.iteration > self.generator_train_start_steps:

            wav_ = self.generator(mel)

            # initialize
            gen_loss = 0.0
            aux_loss = 0.0

            # mel spectrogram loss
            mel_loss = self.criterion_mel(wav_, wav)
            aux_loss += mel_loss
            report("train/mel_loss", float(mel_loss))
            losses_dict["mel_loss"] = float(mel_loss)

            gen_loss += aux_loss * self.lambda_aux

            # adversarial loss
            if self.state.iteration > self.discriminator_train_start_steps:
                p_ = self.discriminator(wav_)
                adv_loss = self.criterion_gen_adv(p_)
                report("train/adversarial_loss", float(adv_loss))
                losses_dict["adversarial_loss"] = float(adv_loss)

                # feature matching loss
                # no need to track gradients
                with paddle.no_grad():
                    p = self.discriminator(wav)
                fm_loss = self.criterion_feat_match(p_, p)
                report("train/feature_matching_loss", float(fm_loss))
                losses_dict["feature_matching_loss"] = float(fm_loss)

                adv_loss += self.lambda_feat_match * fm_loss

                gen_loss += self.lambda_adv * adv_loss

            report("train/generator_loss", float(gen_loss))
            losses_dict["generator_loss"] = float(gen_loss)

            self.optimizer_g.clear_grad()
            gen_loss.backward()

            self.optimizer_g.step()
            self.scheduler_g.step()

        # Disctiminator
        if self.state.iteration > self.discriminator_train_start_steps:
            # re-compute wav_ which leads better quality
            with paddle.no_grad():
                wav_ = self.generator(mel)

            p = self.discriminator(wav)
            p_ = self.discriminator(wav_.detach())
            real_loss, fake_loss = self.criterion_dis_adv(p_, p)
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

        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        '''


class StarGANv2VCEvaluator(StandardEvaluator):
    def __init__(self,
                 models: Dict[str, Layer],
                 criterions: Dict[str, Layer],
                 dataloader: DataLoader,
                 g_loss_params: Dict[str, Any]={
                     'lambda_sty': 1.,
                     'lambda_cyc': 5.,
                     'lambda_ds': 1.,
                     'lambda_norm': 1.,
                     'lambda_asr': 10.,
                     'lambda_f0': 5.,
                     'lambda_f0_sty': 0.1,
                     'lambda_adv': 2.,
                     'lambda_adv_cls': 0.5,
                     'norm_bias': 0.5,
                 },
                 d_loss_params: Dict[str, Any]={
                     'lambda_reg': 1.,
                     'lambda_adv_cls': 0.1,
                     'lambda_con_reg': 10.,
                 },
                 output_dir=None):
        self.models = models
        self.generator = models['generator']
        self.style_encoder = models['style_encoder']
        self.mapping_network = models['mapping_network']
        self.discriminator = models['discriminator']
        self.F0_model = models['F0_model']
        self.asr_model = models['asr_model']

        self.criterions = criterions
        self.criterion_f0 = criterions['f0']
        self.criterion_r1_reg = criterions['r1_reg']
        self.criterion_adv = criterions["adv"]

        self.dataloader = dataloader

        self.g_loss_params = g_loss_params
        self.d_loss_params = d_loss_params

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def evaluate_core(self, batch):
        # logging.debug("Evaluate: ")
        self.msg = "Evaluate: "
        losses_dict = {}
        wav, mel = batch

        # Generator

        wav_ = self.generator(mel)

        # initialize
        gen_loss = 0.0
        aux_loss = 0.0

        ## Adversarial loss
        p_ = self.discriminator(wav_)
        adv_loss = self.criterion_gen_adv(p_)
        report("eval/adversarial_loss", float(adv_loss))
        losses_dict["adversarial_loss"] = float(adv_loss)

        # feature matching loss
        p = self.discriminator(wav)
        fm_loss = self.criterion_feat_match(p_, p)
        report("eval/feature_matching_loss", float(fm_loss))
        losses_dict["feature_matching_loss"] = float(fm_loss)
        adv_loss += self.lambda_feat_match * fm_loss

        gen_loss += self.lambda_adv * adv_loss

        # mel spectrogram loss
        mel_loss = self.criterion_mel(wav_, wav)
        aux_loss += mel_loss
        report("eval/mel_loss", float(mel_loss))
        losses_dict["mel_loss"] = float(mel_loss)

        gen_loss += aux_loss * self.lambda_aux

        report("eval/generator_loss", float(gen_loss))
        losses_dict["generator_loss"] = float(gen_loss)

        # Disctiminator
        p = self.discriminator(wav)
        real_loss, fake_loss = self.criterion_dis_adv(p_, p)
        dis_loss = real_loss + fake_loss
        report("eval/real_loss", float(real_loss))
        report("eval/fake_loss", float(fake_loss))
        report("eval/discriminator_loss", float(dis_loss))

        losses_dict["real_loss"] = float(real_loss)
        losses_dict["fake_loss"] = float(fake_loss)
        losses_dict["discriminator_loss"] = float(dis_loss)

        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        self.logger.info(self.msg)
