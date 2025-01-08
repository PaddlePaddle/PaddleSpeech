# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#
# This file contains code derived from [Descript]
# (https://github.com/descriptinc/audiotools),
# which is licensed under the MIT License:
#
# MIT License
#
# Copyright (c) 2023-Present, Descript
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import typing

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddle.io import SequenceSampler


class ResumableDistributedSampler(DistributedBatchSampler):  # pragma: no cover
    """Distributed sampler that can be resumed from a given start index."""

    def __init__(self, dataset, start_idx: int=None, **kwargs):
        super().__init__(dataset, **kwargs)
        # Start index, allows to resume an experiment at the index it was
        self.start_idx = start_idx // self.num_replicas if start_idx is not None else 0

    def __iter__(self):
        for i, idx in enumerate(super().__iter__()):
            if i >= self.start_idx:
                yield idx
        self.start_idx = 0  # set the index back to 0 so for the next epoch


class ResumableSequentialSampler(SequenceSampler):  # pragma: no cover
    """Sequential sampler that can be resumed from a given start index."""

    def __init__(self, dataset, start_idx: int=None, **kwargs):
        super().__init__(dataset, **kwargs)
        # Start index, allows to resume an experiment at the index it was
        self.start_idx = start_idx if start_idx is not None else 0

    def __iter__(self):
        for i, idx in enumerate(super().__iter__()):
            if i >= self.start_idx:
                yield idx
        self.start_idx = 0  # set the index back to 0 so for the next epoch


class Accelerator:  # pragma: no cover
    """This class is used to prepare models and dataloaders for
    usage with DDP or DP. Use the functions prepare_model, prepare_dataloader to
    prepare the respective objects. In the case of models, they are moved to
    the appropriate GPU. In the case of
    dataloaders, a sampler is created and the dataloader is initialized with
    that sampler.

    If the world size is 1, prepare_model and prepare_dataloader are
    no-ops. If the environment variable ``PADDLE_TRAINER_ID`` is not set, then the
    script was launched without ``paddle.distributed.launch``, and ``DataParallel``
    will be used instead of ``DistributedDataParallel`` (not recommended), if
    the world size (number of GPUs) is greater than 1.

    Parameters
    ----------
    amp : bool, optional
        Whether or not to enable automatic mixed precision, by default False
        (Note: This is a placeholder as PaddlePaddle doesn't have native support for AMP as of now)
    """

    def __init__(self, amp: bool=False):
        trainer_id = os.getenv("PADDLE_TRAINER_ID", None)
        self.world_size = paddle.distributed.get_world_size()

        self.use_ddp = self.world_size > 1 and trainer_id is not None
        self.use_dp = self.world_size > 1 and trainer_id is None
        self.device = "cpu" if self.world_size == 0 else "cuda"

        if self.use_ddp:
            trainer_id = int(trainer_id)
            dist.init_parallel_env()

        self.local_rank = 0 if trainer_id is None else int(trainer_id)
        self.amp = amp

        class DummyScaler:
            def __init__(self):
                pass

            def step(self, optimizer):
                optimizer.step()

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                return optimizer

            def update(self):
                pass

        self.scaler = paddle.amp.GradScaler() if self.amp else DummyScaler()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def prepare_model(self, model: paddle.nn.Layer, **kwargs):
        """Prepares model for DDP or DP. The model is moved to
        the device of the correct rank.

        Parameters
        ----------
        model : paddle.nn.Layer
            Model that is converted for DDP or DP.

        Returns
        -------
        paddle.nn.Layer
            Wrapped model, or original model if DDP and DP are turned off.
        """
        if self.use_ddp:
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = paddle.DataParallel(model, **kwargs)
        elif self.use_dp:
            model = paddle.DataParallel(model, **kwargs)
        return model

    def autocast(self, *args, **kwargs):
        return paddle.amp.auto_cast(self.amp, *args, **kwargs)

    def backward(self, loss: paddle.Tensor):
        """Backwards pass.

        Parameters
        ----------
        loss : paddle.Tensor
            Loss value.
        """
        scaled = self.scaler.scale(loss)  # scale the loss
        scaled.backward()

    def step(self, optimizer: paddle.optimizer.Optimizer):
        """Steps the optimizer.

        Parameters
        ----------
        optimizer : paddle.optimizer.Optimizer
            Optimizer to step forward.
        """
        self.scaler.step(optimizer)

    def update(self):
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/api/paddle/amp/GradScaler_cn.html#step-optimizer
        self.scaler.update()

    def prepare_dataloader(self,
                           dataset: typing.Iterable,
                           start_idx: int=None,
                           **kwargs):
        """Wraps a dataset with a DataLoader, using the correct sampler if DDP is
        enabled.

        Parameters
        ----------
        dataset : typing.Iterable
            Dataset to build Dataloader around.
        start_idx : int, optional
            Start index of sampler, useful if resuming from some epoch,
            by default None

        Returns
        -------
        DataLoader
            Wrapped DataLoader.
        """

        if self.use_ddp:
            sampler = ResumableDistributedSampler(
                dataset,
                start_idx,
                batch_size=kwargs.get("batch_size", 1),
                shuffle=kwargs.get("shuffle", True),
                drop_last=kwargs.get("drop_last", False),
                num_replicas=self.world_size,
                rank=self.local_rank, )
            if "num_workers" in kwargs:
                kwargs["num_workers"] = max(kwargs["num_workers"] //
                                            self.world_size, 1)
        else:
            sampler = ResumableSequentialSampler(dataset, start_idx)

        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler if self.use_ddp else None,
            sampler=sampler if not self.use_ddp else None,
            **kwargs, )
        return dataloader

    @staticmethod
    def unwrap(model):
        return model
