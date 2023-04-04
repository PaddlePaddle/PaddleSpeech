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
from typing import Dict

import paddle
from paddle import distributed as dist
from paddle.io import DataLoader
from paddle.nn import Layer

from . import extension
from ..reporter import DictSummary
from ..reporter import ObsScope
from ..reporter import report
from ..timer import Timer
from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()


class StandardEvaluator(extension.Extension):

    trigger = (1, 'epoch')
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    name = None

    def __init__(self, model: Layer, dataloader: DataLoader):
        # it is designed to hold multiple models
        models = {"main": model}
        self.models: Dict[str, Layer] = models
        self.model = model

        # dataloaders
        self.dataloader = dataloader

    def evaluate_core(self, batch):
        # compute
        self.model(batch)  # you may report here
        return

    def evaluate_sync(self, data):
        # dist sync `evaluate_core` outputs
        if data is None:
            return

        numerator, denominator = data
        if dist.get_world_size() > 1:
            numerator = paddle.to_tensor(numerator)
            denominator = paddle.to_tensor(denominator)
            # the default operator in all_reduce function is sum.
            dist.all_reduce(numerator)
            dist.all_reduce(denominator)
            value = numerator / denominator
            value = float(value)
        else:
            value = numerator / denominator
        # used for `snapshort` to do kbest save.
        report("VALID/LOSS", value)
        logger.info(f"Valid: all-reduce loss {value}")

    def evaluate(self):
        # switch to eval mode
        for model in self.models.values():
            model.eval()

        # to average evaluation metrics
        summary = DictSummary()
        for batch in self.dataloader:
            observation = {}
            with ObsScope(observation):
                # main evaluation computation here.
                with paddle.no_grad():
                    self.evaluate_sync(self.evaluate_core(batch))
            summary.add(observation)
        summary = summary.compute_mean()

        # switch to train mode
        for model in self.models.values():
            model.train()
        return summary

    def __call__(self, trainer=None):
        # evaluate and report the averaged metric to current observation
        # if it is used to extend a trainer, the metrics is reported to
        # to observation of the trainer
        # or otherwise, you can use your own observation
        with Timer("Eval Time Cost: {}"):
            summary = self.evaluate()
        for k, v in summary.items():
            report(k, v)
