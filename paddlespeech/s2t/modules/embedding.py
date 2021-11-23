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
"""Positonal Encoding Module."""
import math
from typing import Tuple

import paddle
from paddle import nn

from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = [
    "PositionalEncodingInterface", "NoPositionalEncoding", "PositionalEncoding",
    "RelPositionalEncoding"
]


class PositionalEncodingInterface:
    def forward(self, x: paddle.Tensor,
                offset: int=0) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute positional encoding.
        Args:
            x (paddle.Tensor): Input tensor (batch, time, `*`).
        Returns:
            paddle.Tensor: Encoded tensor (batch, time, `*`).
            paddle.Tensor: Positional embedding tensor (1, time, `*`).
        """
        raise NotImplementedError("forward method is not implemented")

    def position_encoding(self, offset: int, size: int) -> paddle.Tensor:
        """ For getting encoding in a streaming fashion
        Args:
            offset (int): start offset
            size (int): requried size of position encoding
        Returns:
            paddle.Tensor: Corresponding position encoding
        """
        raise NotImplementedError("position_encoding method is not implemented")


class NoPositionalEncoding(nn.Layer, PositionalEncodingInterface):
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int=5000,
                 reverse: bool=False):
        nn.Layer.__init__(self)

    def forward(self, x: paddle.Tensor,
                offset: int=0) -> Tuple[paddle.Tensor, paddle.Tensor]:
        return x, None

    def position_encoding(self, offset: int, size: int) -> paddle.Tensor:
        return None


class PositionalEncoding(nn.Layer, PositionalEncodingInterface):
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int=5000,
                 reverse: bool=False):
        """Positional encoding.
            PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
            PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
        Args:
            d_model (int): embedding dim.
            dropout_rate (float): dropout rate.
            max_len (int, optional): maximum input length. Defaults to 5000.
            reverse (bool, optional): Not used. Defaults to False.
        """
        nn.Layer.__init__(self)
        self.d_model = d_model
        self.max_len = max_len
        self.xscale = paddle.to_tensor(math.sqrt(self.d_model))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = paddle.zeros([self.max_len, self.d_model])  #[T,D]

        position = paddle.arange(
            0, self.max_len, dtype=paddle.float32).unsqueeze(1)  #[T, 1]
        div_term = paddle.exp(
            paddle.arange(0, self.d_model, 2, dtype=paddle.float32) *
            -(math.log(10000.0) / self.d_model))

        self.pe[:, 0::2] = paddle.sin(position * div_term)
        self.pe[:, 1::2] = paddle.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  #[1, T, D]

    def forward(self, x: paddle.Tensor,
                offset: int=0) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Add positional encoding.
        Args:
            x (paddle.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset
        Returns:
            paddle.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            paddle.Tensor: for compatibility to RelPositionalEncoding, (batch=1, time, ...)
        """
        T = x.shape[1]
        assert offset + x.shape[1] < self.max_len
        #TODO(Hui Zhang): using T = x.size(1), __getitem__ not support Tensor
        pos_emb = self.pe[:, offset:offset + T]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int) -> paddle.Tensor:
        """ For getting encoding in a streaming fashion
        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.
        Args:
            offset (int): start offset
            size (int): requried size of position encoding
        Returns:
            paddle.Tensor: Corresponding position encoding
        """
        assert offset + size < self.max_len
        return self.dropout(self.pe[:, offset:offset + size])


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int=5000):
        """
        Args:
            d_model (int): Embedding dimension.
            dropout_rate (float): Dropout rate.
            max_len (int, optional): [Maximum input length.]. Defaults to 5000.
        """
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x: paddle.Tensor,
                offset: int=0) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute positional encoding.
        Args:
            x (paddle.Tensor): Input tensor (batch, time, `*`).
        Returns:
            paddle.Tensor: Encoded tensor (batch, time, `*`).
            paddle.Tensor: Positional embedding tensor (1, time, `*`).
        """
        assert offset + x.shape[1] < self.max_len
        x = x * self.xscale
        #TODO(Hui Zhang): using x.size(1), __getitem__ not support Tensor
        pos_emb = self.pe[:, offset:offset + x.shape[1]]
        return self.dropout(x), self.dropout(pos_emb)
