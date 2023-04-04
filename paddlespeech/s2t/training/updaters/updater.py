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
from dataclasses import dataclass

import paddle

from paddlespeech.s2t.utils.log import Log

__all__ = ["UpdaterBase", "UpdaterState"]

logger = Log(__name__).getlog()


@dataclass
class UpdaterState:
    iteration: int = 0
    epoch: int = 0


class UpdaterBase():
    """An updater is the abstraction of how a model is trained given the
    dataloader and the optimizer.
    The `update_core` method is a step in the training loop with only necessary
    operations (get a batch, forward and backward, update the parameters).
    Other stuffs are made extensions. Visualization, saving, loading and
    periodical validation and evaluation are not considered here.
    But even in such simplist case, things are not that simple. There is an
    attempt to standardize this process and requires only the model and
    dataset and do all the stuffs automatically. But this may hurt flexibility.
    If we assume a batch yield from the dataloader is just the input to the
    model, we will find that some model requires more arguments, or just some
    keyword arguments. But this prevents us from over-simplifying it.
    From another perspective, the batch may includes not just the input, but
    also the target. But the model's forward method may just need the input.
    We can pass a dict or a super-long tuple to the model and let it pick what
    it really needs. But this is an abuse of lazy interface.
    After all, we care about how a model is trained. But just how the model is
    used for inference. We want to control how a model is trained. We just
    don't want to be messed up with other auxiliary code.
    So the best practice is to define a model and define a updater for it.
    """
    def __init__(self, init_state=None):
        # init state
        if init_state is None:
            self.state = UpdaterState()
        else:
            self.state = init_state

    def update(self, batch):
        raise NotImplementedError(
            "Implement your own `update` method for training a step.")

    def state_dict(self):
        state_dict = {
            "epoch": self.state.epoch,
            "iteration": self.state.iteration,
        }
        return state_dict

    def set_state_dict(self, state_dict):
        self.state.epoch = state_dict["epoch"]
        self.state.iteration = state_dict["iteration"]

    def save(self, path):
        logger.debug(f"Saving to {path}.")
        archive = self.state_dict()
        paddle.save(archive, str(path))

    def load(self, path):
        logger.debug(f"Loading from {path}.")
        archive = paddle.load(str(path))
        self.set_state_dict(archive)
