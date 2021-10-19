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
"""Positional Encoding Module."""
import math

import paddle
from paddle import nn


class PositionalEncoding(nn.Layer):
    """Positional encoding.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    dropout_rate : float
        Dropout rate.
    max_len : int
        Maximum input length.
    reverse : bool
        Whether to reverse the input position.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(paddle.expand(paddle.to_tensor(0.0), (1, max_len)))

    def extend_pe(self, x):
        """Reset the positional encodings."""

        pe = paddle.zeros([x.shape[1], self.d_model])
        if self.reverse:
            position = paddle.arange(
                x.shape[1] - 1, -1, -1.0, dtype=paddle.float32).unsqueeze(1)
        else:
            position = paddle.arange(
                0, x.shape[1], dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(
            paddle.arange(0, self.d_model, 2, dtype=paddle.float32) *
            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x: paddle.Tensor):
        """Add positional encoding.

        Parameters
        ----------
        x : paddle.Tensor
            Input tensor (batch, time, `*`).

        Returns
        ----------
        paddle.Tensor
            Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.

    See Sec. 3.2  https://arxiv.org/abs/1809.08895

    Parameters
    ----------
        d_model : int
            Embedding dimension.
        dropout_rate : float
            Dropout rate.
        max_len : int
            Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(
            d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        x = paddle.ones([1], dtype="float32")
        self.alpha = paddle.create_parameter(
            shape=x.shape,
            dtype=str(x.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(x))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha = paddle.to_tensor(1.0)

    def forward(self, x):
        """Add positional encoding.

        Parameters
        ----------
            x : paddle.Tensor
                Input tensor (batch, time, `*`).

        Returns
        ----------
            paddle.Tensor
                Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, :x.shape[1]]
        return self.dropout(x)
