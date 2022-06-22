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
# Modified from espnet(https://github.com/espnet/espnet)
import numpy


class ChannelSelector():
    """Select 1ch from multi-channel signal"""

    def __init__(self, train_channel="random", eval_channel=0, axis=1):
        self.train_channel = train_channel
        self.eval_channel = eval_channel
        self.axis = axis

    def __repr__(self):
        return ("{name}(train_channel={train_channel}, "
                "eval_channel={eval_channel}, axis={axis})".format(
                    name=self.__class__.__name__,
                    train_channel=self.train_channel,
                    eval_channel=self.eval_channel,
                    axis=self.axis, ))

    def __call__(self, x, train=True):
        # Assuming x: [Time, Channel] by default

        if x.ndim <= self.axis:
            # If the dimension is insufficient, then unsqueeze
            # (e.g [Time] -> [Time, 1])
            ind = tuple(
                slice(None) if i < x.ndim else None
                for i in range(self.axis + 1))
            x = x[ind]

        if train:
            channel = self.train_channel
        else:
            channel = self.eval_channel

        if channel == "random":
            ch = numpy.random.randint(0, x.shape[self.axis])
        else:
            ch = channel

        ind = tuple(
            slice(None) if i != self.axis else ch for i in range(x.ndim))
        return x[ind]
