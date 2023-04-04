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
# Reference chainer MIT (https://opensource.org/licenses/MIT)


class IntervalTrigger():
    """A Predicate to do something every N cycle."""
    def __init__(self, period: int, unit: str):
        if unit not in ("iteration", "epoch"):
            raise ValueError("unit should be 'iteration' or 'epoch'")
        if period <= 0:
            raise ValueError("period should be a positive integer.")
        self.period = period
        self.unit = unit
        self.last_index = None

    def __call__(self, trainer):
        if self.last_index is None:
            last_index = getattr(trainer.updater.state, self.unit)
            self.last_index = last_index

        last_index = self.last_index
        index = getattr(trainer.updater.state, self.unit)
        fire = index // self.period != last_index // self.period

        self.last_index = index
        return fire
