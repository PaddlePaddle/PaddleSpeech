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
from nara_wpe.wpe import wpe


class WPE(object):
    def __init__(self,
                 taps=10,
                 delay=3,
                 iterations=3,
                 psd_context=0,
                 statistics_mode="full"):
        self.taps = taps
        self.delay = delay
        self.iterations = iterations
        self.psd_context = psd_context
        self.statistics_mode = statistics_mode

    def __repr__(self):
        return ("{name}(taps={taps}, delay={delay}"
                "iterations={iterations}, psd_context={psd_context}, "
                "statistics_mode={statistics_mode})".format(
                    name=self.__class__.__name__,
                    taps=self.taps,
                    delay=self.delay,
                    iterations=self.iterations,
                    psd_context=self.psd_context,
                    statistics_mode=self.statistics_mode,
                ))

    def __call__(self, xs):
        """Return enhanced

        :param np.ndarray xs: (Time, Channel, Frequency)
        :return: enhanced_xs
        :rtype: np.ndarray

        """
        # nara_wpe.wpe: (F, C, T)
        xs = wpe(
            xs.transpose((2, 1, 0)),
            taps=self.taps,
            delay=self.delay,
            iterations=self.iterations,
            psd_context=self.psd_context,
            statistics_mode=self.statistics_mode,
        )
        return xs.transpose(2, 1, 0)
