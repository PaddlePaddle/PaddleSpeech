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


class LimitTrigger(object):
    """A Predicate to decide whether to stop."""

    def __init__(self, limit: int, unit: str):
        if unit not in ("iteration", "epoch"):
            raise ValueError("unit should be 'iteration' or 'epoch'")
        if limit <= 0:
            raise ValueError("limit should be a positive integer.")
        self.limit = limit
        self.unit = unit

    def __call__(self, trainer):
        state = trainer.updater.state
        index = getattr(state, self.unit)
        fire = index >= self.limit
        return fire
