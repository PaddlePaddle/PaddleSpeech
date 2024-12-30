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
"""Positional Encoding Module."""
import math

import paddle
from paddle import nn


class PositionalEncoding(nn.Layer):
    """Positional encoding.

    Args:
        d_model (int):
            Embedding dimension.
        dropout_rate (float): 
            Dropout rate.
        max_len (int): 
            Maximum input length.
        reverse (bool): 
            Whether to reverse the input position.
        type (str): 
            dtype of param
    """

    def __init__(self,
                 d_model,
                 dropout_rate,
                 max_len=5000,
                 dtype="float32",
                 reverse=False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.dtype = dtype
        self.extend_pe(paddle.expand(paddle.zeros([1]), (1, max_len)))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        x_shape = paddle.shape(x)
        pe = paddle.zeros([x_shape[1], self.d_model])
        if self.reverse:
            position = paddle.arange(
                x_shape[1] - 1, -1, -1.0, dtype=self.dtype).unsqueeze(1)
        else:
            position = paddle.arange(
                0, x_shape[1], dtype=self.dtype).unsqueeze(1)
        div_term = paddle.exp(
            paddle.arange(0, self.d_model, 2, dtype=self.dtype) *
            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = paddle.assign(pe)

    def forward(self, x: paddle.Tensor):
        """Add positional encoding.

        Args:
            x (Tensor): 
                Input tensor (batch, time, `*`).

        Returns:
            Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        T = paddle.shape(x)[1]
        x = x * self.xscale + self.pe[:, :T]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.
    See Sec. 3.2  https://arxiv.org/abs/1809.08895

    Args:
        d_model (int): 
            Embedding dimension.
        dropout_rate (float): 
            Dropout rate.
        max_len (int): 
            Maximum input length.
        dtype (str): 
            dtype of param
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, dtype="float32"):
        """Initialize class."""
        super().__init__(
            d_model=d_model,
            dropout_rate=dropout_rate,
            max_len=max_len,
            dtype=dtype)
        x = paddle.ones([1], dtype=self.dtype)
        self.alpha = paddle.create_parameter(
            shape=x.shape,
            dtype=self.dtype,
            default_initializer=nn.initializer.Assign(x))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha = paddle.ones([1])

    def forward(self, x):
        """Add positional encoding.

        Args:
            x (Tensor): 
                Input tensor (batch, time, `*`).
        Returns:
            Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        T = paddle.shape(x)[1]
        x = x + self.alpha * self.pe[:, :T]
        return self.dropout(x)


class RelPositionalEncoding(nn.Layer):
    """Relative positional encoding module (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): 
            Embedding dimension.
        dropout_rate (float): 
            Dropout rate.
        max_len (int): 
            Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, dtype="float32"):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.dtype = dtype
        self.extend_pe(paddle.expand(paddle.zeros([1]), (1, max_len)))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if paddle.shape(self.pe)[1] >= paddle.shape(x)[1] * 2 - 1:
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        x_shape = paddle.shape(x)
        pe_positive = paddle.zeros([x_shape[1], self.d_model])
        pe_negative = paddle.zeros([x_shape[1], self.d_model])
        position = paddle.arange(0, x_shape[1], dtype=self.dtype).unsqueeze(1)
        div_term = paddle.exp(
            paddle.arange(0, self.d_model, 2, dtype=self.dtype) *
            -(math.log(10000.0) / self.d_model))
        pe_positive[:, 0::2] = paddle.sin(position * div_term)
        pe_positive[:, 1::2] = paddle.cos(position * div_term)
        pe_negative[:, 0::2] = paddle.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = paddle.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = paddle.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = paddle.concat([pe_positive, pe_negative], axis=1)
        self.pe = pe

    def forward(self, x: paddle.Tensor):
        """Add positional encoding.
        Args:
            x (Tensor):
                Input tensor (batch, time, `*`).
        Returns:
            Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        T = paddle.shape(x)[1]
        pe_size = paddle.shape(self.pe)
        tmp = paddle.cast(paddle.floor(pe_size[1] / 2), dtype='int32')
        pos_emb = self.pe[:, tmp - T + 1:tmp + T, ]
        return self.dropout(x), self.dropout(pos_emb)


class LegacyRelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): 
            Embedding dimension.
        dropout_rate (float): 
            Dropout rate.
        max_len (int): 
            Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int=5000):
        """
        Args:
            d_model (int): 
                Embedding dimension.
            dropout_rate (float): 
                Dropout rate.
            max_len (int, optional): 
                [Maximum input length.]. Defaults to 5000.
        """
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if paddle.shape(self.pe)[1] >= paddle.shape(x)[1]:
                return
        pe = paddle.zeros((paddle.shape(x)[1], self.d_model))
        if self.reverse:
            position = paddle.arange(
                paddle.shape(x)[1] - 1, -1, -1.0,
                dtype=paddle.float32).unsqueeze(1)
        else:
            position = paddle.arange(
                0, paddle.shape(x)[1], dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(
            paddle.arange(0, self.d_model, 2, dtype=paddle.float32) *
            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x: paddle.Tensor):
        """Compute positional encoding.
        Args:
            x (Tensor): 
                Input tensor (batch, time, `*`).
        Returns:
            Tensor: 
                Encoded tensor (batch, time, `*`).
            Tensor: 
                Positional embedding tensor (1, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, :paddle.shape(x)[1]]
        return self.dropout(x), self.dropout(pos_emb)
