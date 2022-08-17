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
# Modified from chainer(https://github.com/chainer/chainer)
import logging
import time
from typing import Dict
from typing import Optional

from paddle import Tensor
from paddle.io import DataLoader
from paddlespeech.t2s.datasets.sampler import DistributedBatchSampler
from paddle.nn import Layer
from paddle.optimizer import Optimizer
from timer import timer

from paddlespeech.t2s.training.reporter import report
from paddlespeech.t2s.training.updater import UpdaterBase
from paddlespeech.t2s.training.updater import UpdaterState


class StandardUpdater(UpdaterBase):
    """An example of over-simplification. Things may not be that simple, but
    you can subclass it to fit your need.
    """

    def __init__(self,
                 model: Layer,
                 optimizer: Optimizer,
                 dataloader: DataLoader,
                 init_state: Optional[UpdaterState]=None):
        # it is designed to hold multiple models
        models = {"main": model}
        self.models: Dict[str, Layer] = models
        self.model = model

        # it is designed to hold multiple optimizers
        optimizers = {"main": optimizer}
        self.optimizer = optimizer
        self.optimizers: Dict[str, Optimizer] = optimizers

        # dataloaders
        self.dataloader = dataloader

        # init state
        if init_state is None:
            self.state = UpdaterState()
        else:
            self.state = init_state

        self.train_iterator = iter(dataloader)
        self.batch_read_time = 0
        self.batch_time = 0

    def update(self):
        # We increase the iteration index after updating and before extension.
        # Here are the reasons.

        # 0. Snapshotting(as well as other extensions, like visualizer) is
        #    executed after a step of updating;
        # 1. We decide to increase the iteration index after updating and
        #    before any all extension is executed. 
        # 3. We do not increase the iteration after extension because we
        #    prefer a consistent resume behavior, when load from a
        #    `snapshot_iter_100.pdz` then the next step to train is `101`,
        #    naturally. But if iteration is increased increased after
        #    extension(including snapshot), then, a `snapshot_iter_99` is
        #    loaded. You would need a extra increasing of the iteration idex
        #    before training to avoid another iteration `99`, which has been
        #    done before snapshotting.
        # 4. Thus iteration index represrnts "currently how mant epochs has
        #    been done."
        # NOTE: use report to capture the correctly value. If you want to
        # report the learning rate used for a step, you must report it before
        # the learning rate scheduler's step() has been called. In paddle's
        # convention, we do not use an extension to change the learning rate.
        # so if you want to report it, do it in the updater.

        # Then here comes the next question. When is the proper time to
        # increase the epoch index? Since all extensions are executed after
        # updating, it is the time that after updating is the proper time to
        # increase epoch index.
        # 1. If we increase the epoch index before updating, then an extension
        #    based ot epoch would miss the correct timing. It could only be
        #    triggerd after an extra updating.
        # 2. Theoretically, when an epoch is done, the epoch index should be
        #    increased. So it would be increase after updating.
        # 3. Thus, eppoch index represents "currently how many epochs has been
        #    done." So it starts from 0.

        # switch to training mode
        for layer in self.models.values():
            layer.train()

        # training for a step is implemented here
        time_before_read = time.time()
        batch = self.read_batch()
        time_before_core = time.time()
        self.update_core(batch)
        self.batch_time = time.time() - time_before_core
        self.batch_read_time = time_before_core - time_before_read
        if isinstance(batch, dict):
            self.batch_size = len(list(batch.items())[0][-1])
        # for pwg
        elif isinstance(batch, list):
            self.batch_size = batch[0].shape[0]

        self.state.iteration += 1
        if self.updates_per_epoch is not None:
            if self.state.iteration % self.updates_per_epoch == 0:
                self.state.epoch += 1

    def update_core(self, batch):
        """A simple case for a training step. Basic assumptions are:
        Single model;
        Single optimizer;
        A batch from the dataloader is just the input of the model;
        The model return a single loss, or a dict containing serval losses.
        Parameters updates at every batch, no gradient accumulation.
        """
        loss = self.model(*batch)

        if isinstance(loss, Tensor):
            loss_dict = {"main": loss}
        else:
            # Dict[str, Tensor]
            loss_dict = loss
            if "main" not in loss_dict:
                main_loss = 0
                for loss_item in loss.values():
                    main_loss += loss_item
                loss_dict["main"] = main_loss

        for name, loss_item in loss_dict.items():
            report(name, float(loss_item))

        self.optimizer.clear_gradient()
        loss_dict["main"].backward()
        self.optimizer.update()

    @property
    def updates_per_epoch(self):
        """Number of updater per epoch, determined by the length of the
        dataloader."""
        length_of_dataloader = None
        try:
            length_of_dataloader = len(self.dataloader)
        except TypeError:
            logging.debug("This dataloader has no __len__.")
        finally:
            return length_of_dataloader

    def new_epoch(self):
        """Start a new epoch."""
        # NOTE: all batch sampler for distributed training should
        # subclass DistributedBatchSampler and implement `set_epoch` method
        batch_sampler = self.dataloader.batch_sampler
        if isinstance(batch_sampler, DistributedBatchSampler):
            batch_sampler.set_epoch(self.state.epoch)
            batch_sampler.shuf_inds(self.state.epoch)
        self.train_iterator = iter(self.dataloader)

    def read_batch(self):
        """Read a batch from the data loader, auto renew when data is exhausted."""
        with timer() as t:
            try:
                batch = next(self.train_iterator)
            except StopIteration:
                self.new_epoch()
                batch = next(self.train_iterator)
            logging.debug(
                f"Read a batch takes {t.elapse}s.")  # replace it with logging
        return batch

    def state_dict(self):
        """State dict of a Updater, model, optimizer and updater state are included."""
        state_dict = super().state_dict()
        for name, layer in self.models.items():
            state_dict[f"{name}_params"] = layer.state_dict()
        for name, optim in self.optimizers.items():
            state_dict[f"{name}_optimizer"] = optim.state_dict()
        return state_dict

    def set_state_dict(self, state_dict):
        """Set state dict for a Updater. Parameters of models, states for
        optimizers and UpdaterState are restored."""
        for name, layer in self.models.items():
            layer.set_state_dict(state_dict[f"{name}_params"])
        for name, optim in self.optimizers.items():
            optim.set_state_dict(state_dict[f"{name}_optimizer"])
        super().set_state_dict(state_dict)
