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
"""Layer normalization module."""
import paddle
from paddle import nn


class LayerNorm(nn.LayerNorm):
    """Layer normalization module.
    Args:
        nout (int): 
            Output dim size.
        dim (int): 
            Dimension to be normalized.
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super().__init__(nout)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (Tensor):
                Input tensor.

        Returns: 
            Tensor: Normalized tensor.
        """

        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        else:
            len_dim = len(x.shape)
            if self.dim < 0:
                self.dim = len_dim + self.dim
            assert self.dim >= 0

            orig_perm = list(range(len_dim))
            new_perm = orig_perm[:]
            # Python style item change is not able when converting dygraph to static graph.
            # new_perm[self.dim], new_perm[len_dim -1] = new_perm[len_dim -1], new_perm[self.dim]
            # use C++ style item change here
            temp = new_perm[self.dim]
            new_perm[self.dim] = new_perm[len_dim - 1]
            new_perm[len_dim - 1] = temp

            return paddle.transpose(
                super(LayerNorm, self).forward(paddle.transpose(x, new_perm)),
                new_perm)
