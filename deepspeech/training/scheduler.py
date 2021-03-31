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
from paddle.optimizer.lr import LRScheduler

logger = logging.getLogger(__name__)

__all__ = ["WarmupLR"]


class WarmupLR(LRScheduler):
    """The WarmupLR scheduler
    This scheduler is almost same as NoamLR Scheduler except for following
    difference:
    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    Note that the maximum lr equals to optimizer.lr in this scheduler.
    """

    def __init__(self,
                 warmup_steps: Union[int, float]=25000,
                 learning_rate=1.0,
                 last_epoch=-1,
                 verbose=False):
        assert check_argument_types()
        self.warmup_steps = warmup_steps
        super().__init__(learning_rate, last_epoch, verbose)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return self.base_lr * self.warmup_steps**0.5 * min(
            step_num**-0.5, step_num * self.warmup_steps**-1.5)

    def set_step(self, step: int):
        self.last_epoch = step
