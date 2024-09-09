# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Mobvoi Inc. All Rights Reserved.
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
# Modified from wenet(https://github.com/wenet-e2e/wenet)
import paddle
from paddle import nn

from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = ['GlobalCMVN']


class GlobalCMVN(nn.Layer):
    def __init__(self,
                 mean: paddle.Tensor,
                 istd: paddle.Tensor,
                 norm_var: bool=True):
        """
        Args:
            mean (paddle.Tensor): mean stats
            istd (paddle.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def __repr__(self):
        return ("{name}(mean={mean}, istd={istd}, norm_var={norm_var})".format(
            name=self.__class__.__name__,
            mean=self.mean,
            istd=self.istd,
            norm_var=self.norm_var))

    def forward(self, x: paddle.Tensor):
        """
        Args:
            x (paddle.Tensor): (batch, max_len, feat_dim)
        Returns:
            (paddle.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x
