#!/usr/bin/python3
#! coding:utf-8

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

import sys
import traceback
import paddle
import time
import six
from typing import Union
from typing import List
from typing import Callable
from pathlib import Path

from collections import OrderedDict
from paddlespeech.s2t.utils.log import Log
from paddle import distributed as dist
from paddlespeech.t2s.training.extension import PRIORITY_READER
from paddlespeech.t2s.training.reporter import scope
from paddlespeech.t2s.training.updater import UpdaterBase
from paddlespeech.t2s.training.extension import Extension
from paddlespeech.t2s.training.trigger import get_trigger
from paddlespeech.t2s.training.triggers.limit_trigger import LimitTrigger
from paddlespeech.t2s.training.trainer import _ExtensionEntry
from paddlespeech.t2s.training.trainer import Trainer
from paddlespeech.t2s.utils import profiler
logger = Log(__name__).getlog()

class SIDTrainer(object):
    def __init__(self, 
                updater: UpdaterBase,
                stop_trigger: Callable=None,
                out: Union[str, Path] = 'result',
                extentions: List[Extension] = None,
                profiler_options: str = None,
                iteration = 0):
        """
        Executable an experiment
        ----------------------
        Arugment:
        out: Union[str, Path]
            experiment directory
        """
        super().__init__(updater=updater,
                         stop_trigger=stop_trigger,
                         out=out,
                         extentions=extentions,
                         profiler_options=profiler_options)
        self.iteration = iteration
        
    def train_batch(self, model, optimizer, lr_scheduler, train_loader,
                    classifier, loss_fn, epoch, writer):
        logger.info("cur epoch: {}".format(epoch))
        # 开启训练模式
        model.train()
        data_start_time = time.time()
        for step, batch in enumerate(train_loader):
            # load the batch data
            dataload_time = time.time() - data_start_time

            # forward the model 
            data_start_time = time.time()
            xs_pad, ilens, spk_ids = batch
            xs_pad = paddle.transpose(xs_pad, perm=[0,2,1])
            # logger.info("xs pad: {}".format(xs_pad))
            model_output = model(xs_pad)
            logits = classifier(model_output)
            loss = loss_fn(logits, spk_ids)
            batch_train_time = time.time() - data_start_time
            data_start_time = time.time()

            # gradient update
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()
            gradient_update_time = time.time() - data_start_time

            # logging
            msg = "Randk: {}, ".format(dist.get_rank())
            msg += "epoch: {}, ".format(epoch)
            msg += "step: {}, ".format(step)
            msg += "loss: {}, ".format(loss.item())
            msg += "data load time: {:>.3f}, ".format(dataload_time)
            msg +=  "model forward time: {:>.3f}, ".format(batch_train_time)
            msg += "gradient update time: {:>.3f}, ".format(gradient_update_time)

            self.iteration += 1
            writer.add_scalar(tag="train loss", step=self.iteration, value=loss.item())
            logger.info(msg)
            data_start_time = time.time()

    @paddle.no_grad()
    def valid(self, model, valid_loader, classifier, loss_fn, epoch, writer):
        model.eval()
        num_seen_utts = 0
        # valid_losses = defaultdict(list)
        total_loss = 0.0
        for step, batch in enumerate(valid_loader):
            xs_pad, ilens, spk_ids = batch
            xs_pad = paddle.transpose(xs_pad, perm=[0,2,1])
            model_output = model(xs_pad)
            # logger.info("model output: {}".format(model_output))
            logits = classifier(model_output)
            loss = loss_fn(logits, spk_ids)
            # logger.info("step: {}, loss: {}".format(step, loss.item()))
            if paddle.isfinite(loss):
                num_seen_utts += xs_pad.shape[0]
                total_loss += float(loss.item())

        if dist.get_rank() == 0:
            writer.add_scalar(tag="val loss", step=epoch, value=total_loss)
        
        return total_loss