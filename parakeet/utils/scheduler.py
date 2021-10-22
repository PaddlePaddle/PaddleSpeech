# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ["SchedulerBase", "Constant", "PieceWise", "StepWise"]


class SchedulerBase(object):
    def __call__(self, step):
        raise NotImplementedError("You should implement the __call__ method.")


class Constant(SchedulerBase):
    def __init__(self, value):
        self.value = value

    def __call__(self, step):
        return self.value


class PieceWise(SchedulerBase):
    def __init__(self, anchors):
        anchors = list(anchors)
        anchors = sorted(anchors, key=lambda x: x[0])
        assert anchors[0][0] == 0, "it must start from zero"
        self.xs = [item[0] for item in anchors]
        self.ys = [item[1] for item in anchors]
        self.num_anchors = len(self.xs)

    def __call__(self, step):
        i = 0
        for x in self.xs:
            if step >= x:
                i += 1
        if i == 0:
            return self.ys[0]
        if i == self.num_anchors:
            return self.ys[-1]
        k = (self.ys[i] - self.ys[i - 1]) / (self.xs[i] - self.xs[i - 1])
        out = self.ys[i - 1] + (step - self.xs[i - 1]) * k
        return out


class StepWise(SchedulerBase):
    def __init__(self, anchors):
        anchors = list(anchors)
        anchors = sorted(anchors, key=lambda x: x[0])
        assert anchors[0][0] == 0, "it must start from zero"
        self.xs = [item[0] for item in anchors]
        self.ys = [item[1] for item in anchors]
        self.num_anchors = len(self.xs)

    def __call__(self, step):
        i = 0
        for x in self.xs:
            if step >= x:
                i += 1

        if i == self.num_anchors:
            return self.ys[-1]
        if i == 0:
            return self.ys[0]
        return self.ys[i - 1]
