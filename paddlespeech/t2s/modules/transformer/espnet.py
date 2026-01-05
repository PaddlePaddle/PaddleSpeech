import paddle

"""Positonal Encoding Module."""
import math
from typing import Tuple, Union

import numpy as np


class PositionalEncoding(paddle.nn.Layer):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        max_len: int = 5000,
        reverse: bool = False,
    ):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        self.pe = paddle.zeros(self.max_len, self.d_model)
        position = paddle.arange(0, self.max_len, dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(
            x=paddle.arange(0, self.d_model, 2, dtype=paddle.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        self.pe[:, 0::2] = paddle.sin(position * div_term)
        self.pe[:, 1::2] = paddle.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(
        self, x: paddle.Tensor, offset: Union[int, paddle.Tensor] = 0
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        self.pe = self.pe.to(x.place)
        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(
        self, offset: Union[int, paddle.Tensor], size: int, apply_dropout: bool = True
    ) -> paddle.Tensor:
        """For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        elif isinstance(offset, paddle.Tensor) and offset.dim() == 0:
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        else:
            assert paddle.compat.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + paddle.arange(0, size).to(offset.place)
            flag = index > 0
            index = index * flag
            pos_emb = paddle.nn.functional.embedding(index, self.pe[0])
        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(
        self, x: paddle.Tensor, offset: Union[int, paddle.Tensor] = 0
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.pe = self.pe.to(x.place)
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


class WhisperPositionalEncoding(PositionalEncoding):
    """Sinusoids position encoding used in openai-whisper.encoder"""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 1500):
        super().__init__(d_model, dropout_rate, max_len)
        self.xscale = 1.0
        log_timescale_increment = np.log(10000) / (d_model // 2 - 1)
        inv_timescales = paddle.exp(
            x=-log_timescale_increment * paddle.arange(d_model // 2)
        )
        scaled_time = (
            paddle.arange(max_len)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        pe = paddle.cat([paddle.sin(scaled_time), paddle.cos(scaled_time)], dim=1)
        delattr(self, "pe")
        self.register_buffer(name="pe", tensor=pe.unsqueeze(0))


class LearnablePositionalEncoding(PositionalEncoding):
    """Learnable position encoding used in openai-whisper.decoder"""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 448):
        super().__init__(d_model, dropout_rate, max_len)
        self.pe = paddle.nn.parameter.Parameter(paddle.empty(1, max_len, d_model))
        self.xscale = 1.0


class NoPositionalEncoding(paddle.nn.Layer):
    """No position encoding"""

    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = paddle.nn.Dropout(p=dropout_rate)

    def forward(
        self, x: paddle.Tensor, offset: Union[int, paddle.Tensor] = 0
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Just return zero vector for interface compatibility"""
        pos_emb = paddle.zeros(1, x.size(1), self.d_model).to(x.place)
        return self.dropout(x), pos_emb

    def position_encoding(
        self, offset: Union[int, paddle.Tensor], size: int
    ) -> paddle.Tensor:
        return paddle.zeros(1, size, self.d_model)


class EspnetRelPositionalEncoding(paddle.nn.Layer):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Construct an PositionalEncoding object."""
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(paddle.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: paddle.Tensor):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.place != x.place:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.place)
                return
        pe_positive = paddle.zeros(x.size(1), self.d_model)
        pe_negative = paddle.zeros(x.size(1), self.d_model)
        position = paddle.arange(0, x.size(1), dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(
            x=paddle.arange(0, self.d_model, 2, dtype=paddle.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = paddle.sin(position * div_term)
        pe_positive[:, 1::2] = paddle.cos(position * div_term)
        pe_negative[:, 0::2] = paddle.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = paddle.cos(-1 * position * div_term)
        pe_positive = paddle.flip(x=pe_positive, axis=[0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = paddle.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.place, dtype=x.dtype)

    def forward(
        self, x: paddle.Tensor, offset: Union[int, paddle.Tensor] = 0
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(
        self, offset: Union[int, paddle.Tensor], size: int
    ) -> paddle.Tensor:
        """For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        if isinstance(offset, int):
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2
                - size
                - offset
                + 1 : self.pe.size(1) // 2
                + size
                + offset,
            ]
        elif isinstance(offset, paddle.Tensor):
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2
                - size
                - offset
                + 1 : self.pe.size(1) // 2
                + size
                + offset,
            ]
        return pos_emb
