# Authors
#  * Mirco Ravanelli 2020
#  * Guillermo Cámbara 2021
#  * Sarthak Yadav 2022
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from speechbrain(https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/nnet/normalization.py)
import paddle.nn as nn

from paddlespeech.s2t.modules.align import BatchNorm1D


class BatchNorm1d(nn.Layer):
    """Applies 1d batch normalization to the input tensor.
    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.
    Example
    -------
    >>> input = paddle.randn([100, 10])
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    Paddle.Shape([100, 10])
    """

    def __init__(
            self,
            input_shape=None,
            input_size=None,
            eps=1e-05,
            momentum=0.9,
            combine_batch_time=False,
            skip_transpose=False, ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = BatchNorm1D(input_size, momentum=momentum, epsilon=eps)

    def forward(self, x):
        """Returns the normalized input tensor.
        Arguments
        ---------
        x : paddle.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[3],
                              shape_or[2])

        elif not self.skip_transpose:
            x = x.transpose([0, 2, 1])

        x_n = self.norm(x)
        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose([0, 2, 1])

        return x_n
