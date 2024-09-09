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


class TimeTrigger():
    """Trigger based on a fixed time interval.
    This trigger accepts iterations with a given interval time.
    Args:
        period (float): Interval time. It is given in seconds.
    """

    def __init__(self, period):
        self._period = period
        self._next_time = self._period

    def __call__(self, trainer):
        if self._next_time < trainer.elapsed_time:
            self._next_time += self._period
            return True
        else:
            return False

    def state_dict(self):
        state_dict = {
            "next_time": self._next_time,
        }
        return state_dict

    def set_state_dict(self, state_dict):
        self._next_time = state_dict['next_time']
