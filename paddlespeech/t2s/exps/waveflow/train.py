# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import time

import numpy as np
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

from paddlespeech.t2s.data import dataset
from paddlespeech.t2s.exps.waveflow.config import get_cfg_defaults
from paddlespeech.t2s.exps.waveflow.ljspeech import LJSpeech
from paddlespeech.t2s.exps.waveflow.ljspeech import LJSpeechClipCollector
from paddlespeech.t2s.exps.waveflow.ljspeech import LJSpeechCollector
from paddlespeech.t2s.models.waveflow import ConditionalWaveFlow
from paddlespeech.t2s.models.waveflow import WaveFlowLoss
from paddlespeech.t2s.training.cli import default_argument_parser
from paddlespeech.t2s.training.experiment import ExperimentBase
from paddlespeech.t2s.utils import mp_tools


class Experiment(ExperimentBase):
    def setup_model(self):
        config = self.config
        model = ConditionalWaveFlow(
            upsample_factors=config.model.upsample_factors,
            n_flows=config.model.n_flows,
            n_layers=config.model.n_layers,
            n_group=config.model.n_group,
            channels=config.model.channels,
            n_mels=config.data.n_mels,
            kernel_size=config.model.kernel_size)

        if self.parallel:
            model = paddle.DataParallel(model)
        optimizer = paddle.optimizer.Adam(
            config.training.lr, parameters=model.parameters())
        criterion = WaveFlowLoss(sigma=config.model.sigma)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def setup_dataloader(self):
        config = self.config
        args = self.args

        ljspeech_dataset = LJSpeech(args.data)
        valid_set, train_set = dataset.split(ljspeech_dataset,
                                             config.data.valid_size)

        batch_fn = LJSpeechClipCollector(config.data.clip_frames,
                                         config.data.hop_length)

        if not self.parallel:
            train_loader = DataLoader(
                train_set,
                batch_size=config.data.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=batch_fn)
        else:
            sampler = DistributedBatchSampler(
                train_set,
                batch_size=config.data.batch_size,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=True)
            train_loader = DataLoader(
                train_set, batch_sampler=sampler, collate_fn=batch_fn)

        valid_batch_fn = LJSpeechCollector()
        valid_loader = DataLoader(
            valid_set, batch_size=1, collate_fn=valid_batch_fn)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def compute_outputs(self, mel, wav):
        # model_core = model._layers if isinstance(model, paddle.DataParallel) else model
        z, log_det_jocobian = self.model(wav, mel)
        return z, log_det_jocobian

    def train_batch(self):
        start = time.time()
        batch = self.read_batch()
        data_loader_time = time.time() - start

        self.model.train()
        self.optimizer.clear_grad()
        mel, wav = batch
        z, log_det_jocobian = self.compute_outputs(mel, wav)
        loss = self.criterion(z, log_det_jocobian)
        loss.backward()
        self.optimizer.step()
        iteration_time = time.time() - start

        loss_value = float(loss)
        msg = "Rank: {}, ".format(dist.get_rank())
        msg += "step: {}, ".format(self.iteration)
        msg += "time: {:>.3f}s/{:>.3f}s, ".format(data_loader_time,
                                                  iteration_time)
        msg += "loss: {:>.6f}".format(loss_value)
        self.logger.info(msg)
        if dist.get_rank() == 0:
            self.visualizer.add_scalar("train/loss", loss_value, self.iteration)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        valid_iterator = iter(self.valid_loader)
        valid_losses = []
        mel, wav = next(valid_iterator)
        z, log_det_jocobian = self.compute_outputs(mel, wav)
        loss = self.criterion(z, log_det_jocobian)
        valid_losses.append(float(loss))
        valid_loss = np.mean(valid_losses)
        self.visualizer.add_scalar("valid/loss", valid_loss, self.iteration)


def main_sp(config, args):
    exp = Experiment(config, args)
    exp.setup()
    exp.resume_or_load()
    exp.run()


def main(config, args):
    if args.ngpu > 1:
        dist.spawn(main_sp, args=(config, args), nprocs=args.ngpu)
    else:
        main_sp(config, args)


if __name__ == "__main__":
    config = get_cfg_defaults()
    parser = default_argument_parser()
    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)
