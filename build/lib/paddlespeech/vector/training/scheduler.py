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
from paddle.optimizer.lr import LRScheduler


class CyclicLRScheduler(LRScheduler):
    def __init__(self,
                 base_lr: float=1e-8,
                 max_lr: float=1e-3,
                 step_size: int=10000):

        super(CyclicLRScheduler, self).__init__()

        self.current_step = -1
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size

    def step(self):
        if not hasattr(self, 'current_step'):
            return

        self.current_step += 1
        if self.current_step >= 2 * self.step_size:
            self.current_step %= 2 * self.step_size

        self.last_lr = self.get_lr()

    def get_lr(self):
        p = self.current_step / (2 * self.step_size)  # Proportion in one cycle.
        if p < 0.5:  # Increase
            return self.base_lr + p / 0.5 * (self.max_lr - self.base_lr)
        else:  # Decrease
            return self.max_lr - (p / 0.5 - 1) * (self.max_lr - self.base_lr)
