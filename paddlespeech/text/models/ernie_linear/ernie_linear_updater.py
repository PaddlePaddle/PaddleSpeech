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
import logging

import paddle
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.nn import Layer
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler
from sklearn.metrics import f1_score

from paddlespeech.t2s.training.extensions.evaluator import StandardEvaluator
from paddlespeech.t2s.training.reporter import report
from paddlespeech.t2s.training.updaters.standard_updater import StandardUpdater

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ErnieLinearUpdater(StandardUpdater):
    def __init__(self,
                 model: Layer,
                 criterion: Layer,
                 scheduler: LRScheduler,
                 optimizer: Optimizer,
                 dataloader: DataLoader,
                 output_dir=None):
        super().__init__(model, optimizer, dataloader, init_state=None)
        self.model = model
        self.dataloader = dataloader

        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def update_core(self, batch):
        self.msg = "Rank: {}, ".format(dist.get_rank())
        losses_dict = {}

        input, label = batch
        label = paddle.reshape(label, shape=[-1])
        y, logit = self.model(input)
        pred = paddle.argmax(logit, axis=1)

        loss = self.criterion(y, label)

        self.optimizer.clear_grad()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        F1_score = f1_score(label.numpy().tolist(),
                            pred.numpy().tolist(),
                            average="macro")

        report("train/loss", float(loss))
        losses_dict["loss"] = float(loss)
        report("train/F1_score", float(F1_score))
        losses_dict["F1_score"] = float(F1_score)

        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())


class ErnieLinearEvaluator(StandardEvaluator):
    def __init__(self,
                 model: Layer,
                 criterion: Layer,
                 dataloader: DataLoader,
                 output_dir=None):
        super().__init__(model, dataloader)
        self.model = model
        self.criterion = criterion
        self.dataloader = dataloader

        log_file = output_dir / 'worker_{}.log'.format(dist.get_rank())
        self.filehandler = logging.FileHandler(str(log_file))
        logger.addHandler(self.filehandler)
        self.logger = logger
        self.msg = ""

    def evaluate_core(self, batch):
        self.msg = "Evaluate: "
        losses_dict = {}

        input, label = batch
        label = paddle.reshape(label, shape=[-1])
        y, logit = self.model(input)
        pred = paddle.argmax(logit, axis=1)

        loss = self.criterion(y, label)

        F1_score = f1_score(label.numpy().tolist(),
                            pred.numpy().tolist(),
                            average="macro")

        report("eval/loss", float(loss))
        losses_dict["loss"] = float(loss)
        report("eval/F1_score", float(F1_score))
        losses_dict["F1_score"] = float(F1_score)

        self.msg += ', '.join('{}: {:>.6f}'.format(k, v)
                              for k, v in losses_dict.items())
        self.logger.info(self.msg)
