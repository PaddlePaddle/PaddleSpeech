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

from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.nn import Layer
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler

from paddlespeech.t2s.models.starganv2_vc.losses import compute_d_loss
from paddlespeech.t2s.models.starganv2_vc.losses import compute_g_loss
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
                 adv_cls_epoch: int=50,
                 con_reg_epoch: int=30,
                 use_r1_reg: bool=False,
                 output_dir=None):
        self.models = models

        self.optimizers = optimizers
        self.optimizer_g = optimizers['generator']
        self.optimizer_s = optimizers['style_encoder']
        self.optimizer_m = optimizers['mapping_network']
        self.optimizer_d = optimizers['discriminator']

        self.schedulers = schedulers
        self.scheduler_g = schedulers['generator']
        self.scheduler_s = schedulers['style_encoder']
        self.scheduler_m = schedulers['mapping_network']
        self.scheduler_d = schedulers['discriminator']

        self.dataloader = dataloader

        self.g_loss_params = g_loss_params
        self.d_loss_params = d_loss_params

        self.use_r1_reg = use_r1_reg
        self.con_reg_epoch = con_reg_epoch
        self.adv_cls_epoch = adv_cls_epoch

        self.state = UpdaterState(iteration=0, epoch=0)
        self.train_iterator = iter(self.dataloader)

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def zero_grad(self):
        self.optimizer_d.clear_grad()
        self.optimizer_g.clear_grad()
        self.optimizer_m.clear_grad()
        self.optimizer_s.clear_grad()

    def scheduler(self):
        self.scheduler_d.step()
        self.scheduler_g.step()
        self.scheduler_m.step()
        self.scheduler_s.step()

    def update_core(self, batch):
        self.msg = "Rank: {}, ".format(dist.get_rank())
        losses_dict = {}
        # parse batch
        x_real = batch['x_real']
        y_org = batch['y_org']
        x_ref = batch['x_ref']
        x_ref2 = batch['x_ref2']
        y_trg = batch['y_trg']
        z_trg = batch['z_trg']
        z_trg2 = batch['z_trg2']

        use_con_reg = (self.state.epoch >= self.con_reg_epoch)
        use_adv_cls = (self.state.epoch >= self.adv_cls_epoch)

        # Discriminator loss
        # train the discriminator (by random reference)
        self.zero_grad()
        random_d_loss = compute_d_loss(
            nets=self.models,
            x_real=x_real,
            y_org=y_org,
            y_trg=y_trg,
            z_trg=z_trg,
            use_adv_cls=use_adv_cls,
            use_con_reg=use_con_reg,
            **self.d_loss_params)
        random_d_loss.backward()
        self.optimizer_d.step()
        # train the discriminator (by target reference)
        self.zero_grad()
        target_d_loss = compute_d_loss(
            nets=self.models,
            x_real=x_real,
            y_org=y_org,
            y_trg=y_trg,
            x_ref=x_ref,
            use_adv_cls=use_adv_cls,
            use_con_reg=use_con_reg,
            **self.d_loss_params)
        target_d_loss.backward()
        self.optimizer_d.step()
        report("train/random_d_loss", float(random_d_loss))
        report("train/target_d_loss", float(target_d_loss))
        losses_dict["random_d_loss"] = float(random_d_loss)
        losses_dict["target_d_loss"] = float(target_d_loss)

        # Generator
        # train the generator (by random reference)
        self.zero_grad()
        random_g_loss = compute_g_loss(
            nets=self.models,
            x_real=x_real,
            y_org=y_org,
            y_trg=y_trg,
            z_trgs=[z_trg, z_trg2],
            use_adv_cls=use_adv_cls,
            **self.g_loss_params)
        random_g_loss.backward()
        self.optimizer_g.step()
        self.optimizer_m.step()
        self.optimizer_s.step()

        # train the generator (by target reference)
        self.zero_grad()
        target_g_loss = compute_g_loss(
            nets=self.models,
            x_real=x_real,
            y_org=y_org,
            y_trg=y_trg,
            x_refs=[x_ref, x_ref2],
            use_adv_cls=use_adv_cls,
            **self.g_loss_params)
        target_g_loss.backward()
        # 此处是否要 optimizer_g optimizer_m optimizer_s 都写上？
        # 源码没写上后两个是否是疏忽？
        self.optimizer_g.step()
        # self.optimizer_m.step()
        # self.optimizer_s.step()
        report("train/random_g_loss", float(random_g_loss))
        report("train/target_g_loss", float(target_g_loss))
        losses_dict["random_g_loss"] = float(random_g_loss)
        losses_dict["target_g_loss"] = float(target_g_loss)

        self.scheduler()

        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())


class StarGANv2VCEvaluator(StandardEvaluator):
    def __init__(self,
                 models: Dict[str, Layer],
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
                 adv_cls_epoch: int=50,
                 con_reg_epoch: int=30,
                 use_r1_reg: bool=False,
                 output_dir=None):
        self.models = models

        self.dataloader = dataloader

        self.g_loss_params = g_loss_params
        self.d_loss_params = d_loss_params

        self.use_r1_reg = use_r1_reg
        self.con_reg_epoch = con_reg_epoch
        self.adv_cls_epoch = adv_cls_epoch

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def evaluate_core(self, batch):
        # logging.debug("Evaluate: ")
        self.msg = "Evaluate: "
        losses_dict = {}

        x_real = batch['x_real']
        y_org = batch['y_org']
        x_ref = batch['x_ref']
        x_ref2 = batch['x_ref2']
        y_trg = batch['y_trg']
        z_trg = batch['z_trg']
        z_trg2 = batch['z_trg2']

        # eval the discriminator

        random_d_loss = compute_d_loss(
            nets=self.models,
            x_real=x_real,
            y_org=y_org,
            y_trg=y_trg,
            z_trg=z_trg,
            use_r1_reg=self.use_r1_reg,
            use_adv_cls=use_adv_cls,
            **self.d_loss_params)

        target_d_loss = compute_d_loss(
            nets=self.models,
            x_real=x_real,
            y_org=y_org,
            y_trg=y_trg,
            x_ref=x_ref,
            use_r1_reg=self.use_r1_reg,
            use_adv_cls=use_adv_cls,
            **self.d_loss_params)

        report("eval/random_d_loss", float(random_d_loss))
        report("eval/target_d_loss", float(target_d_loss))
        losses_dict["random_d_loss"] = float(random_d_loss)
        losses_dict["target_d_loss"] = float(target_d_loss)

        # eval the generator

        random_g_loss = compute_g_loss(
            nets=self.models,
            x_real=x_real,
            y_org=y_org,
            y_trg=y_trg,
            z_trgs=[z_trg, z_trg2],
            use_adv_cls=use_adv_cls,
            **self.g_loss_params)

        target_g_loss = compute_g_loss(
            nets=self.models,
            x_real=x_real,
            y_org=y_org,
            y_trg=y_trg,
            x_refs=[x_ref, x_ref2],
            use_adv_cls=use_adv_cls,
            **self.g_loss_params)

        report("eval/random_g_loss", float(random_g_loss))
        report("eval/target_g_loss", float(target_g_loss))
        losses_dict["random_g_loss"] = float(random_g_loss)
        losses_dict["target_g_loss"] = float(target_g_loss)

        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        self.logger.info(self.msg)
