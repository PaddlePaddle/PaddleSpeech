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
# Conv2dSubsampling 测试通过
"""Subsampling layer definition."""
import paddle

from paddlespeech.t2s.modules.transformer.embedding import PositionalEncoding


class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.
    Parameters
    ----------
    message : str
        Message for error catch
    actual_size : int
        the short size that cannot pass the subsampling
    limit : int
        the limit size for subsampling
    """

    def __init__(self, message, actual_size, limit):
        """Construct a TooShortUttError for error handler."""
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(ins, size):
    """Check if the utterance is too short for subsampling."""
    if isinstance(ins, Conv2dSubsampling2) and size < 3:
        return True, 3
    if isinstance(ins, Conv2dSubsampling) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling6) and size < 11:
        return True, 11
    if isinstance(ins, Conv2dSubsampling8) and size < 15:
        return True, 15
    return False, -1


class Conv2dSubsampling(paddle.nn.Layer):
    """Convolutional 2D subsampling (to 1/4 length).
    Parameters
    ----------
    idim : int
        Input dimension.
    odim : int
        Output dimension.
    dropout_rate : float
        Dropout rate.
    pos_enc : paddle.nn.Layer
        Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(1, odim, 3, 2),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(odim, odim, 3, 2),
            paddle.nn.ReLU(), )
        self.out = paddle.nn.Sequential(
            paddle.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else
            PositionalEncoding(odim, dropout_rate), )

    def forward(self, x, x_mask):
        """Subsample x.
        Parameters
        ----------
        x : paddle.Tensor
            Input tensor (#batch, time, idim).
        x_mask : paddle.Tensor
            Input mask (#batch, 1, time).
        Returns
        ----------
        paddle.Tensor
            Subsampled tensor (#batch, time', odim),
            where time' = time // 4.
        paddle.Tensor
            Subsampled mask (#batch, 1, time'),
            where time' = time // 4.
        """
        # (b, c, t, f)
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.shape
        # x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x = self.out(x.transpose([0, 2, 1, 3]).reshape([b, t, c * f]))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.
        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.
        """
        if key != -1:
            raise NotImplementedError(
                "Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling2(paddle.nn.Layer):
    """Convolutional 2D subsampling (to 1/2 length).
    Parameters
    ----------
    idim : int
        Input dimension.
    odim : int
        Output dimension.
    dropout_rate : float
        Dropout rate.
    pos_enc : paddle.nn.Layer
        Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(1, odim, 3, 2),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(odim, odim, 3, 1),
            paddle.nn.ReLU(), )
        self.out = paddle.nn.Sequential(
            paddle.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim),
            pos_enc if pos_enc is not None else
            PositionalEncoding(odim, dropout_rate), )

    def forward(self, x, x_mask):
        """Subsample x.
        Parameters
        ----------
        x : paddle.Tensor
            Input tensor (#batch, time, idim).
        x_mask : paddle.Tensor
            Input mask (#batch, 1, time).
        Returns
        ----------
        paddle.Tensor
            ubsampled tensor (#batch, time', odim),
            where time' = time // 2.
        paddle.Tensor
            Subsampled mask (#batch, 1, time'),
            where time' = time // 2.
        """
        # (b, c, t, f)
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.out(x.transpose([0, 2, 1, 3]).reshape([b, t, c * f]))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:1]

    def __getitem__(self, key):
        """Get item.
        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.
        """
        if key != -1:
            raise NotImplementedError(
                "Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling6(paddle.nn.Layer):
    """Convolutional 2D subsampling (to 1/6 length).
    Parameters
    ----------
    idim : int
        Input dimension.
    odim : int
        Output dimension.
    dropout_rate : float
        Dropout rate.
    pos_enc : paddle.nn.Layer
        Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(1, odim, 3, 2),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(odim, odim, 5, 3),
            paddle.nn.ReLU(), )
        self.out = paddle.nn.Sequential(
            paddle.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim),
            pos_enc if pos_enc is not None else
            PositionalEncoding(odim, dropout_rate), )

    def forward(self, x, x_mask):
        """Subsample x.
        Parameters
        ----------
        x : paddle.Tensor
            Input tensor (#batch, time, idim).
        x_mask paddle.Tensor
            Input mask (#batch, 1, time).
        Returns
        ----------
        paddle.Tensor
            Subsampled tensor (#batch, time', odim),
            where time' = time // 6.
        paddle.Tensor
            Subsampled mask (#batch, 1, time'),
            where time' = time // 6.
        """
        # (b, c, t, f)
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.out(x.transpose([0, 2, 1, 3]).reshape([b, t, c * f]))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(paddle.nn.Layer):
    """Convolutional 2D subsampling (to 1/8 length).
    Parameters
    ----------
    idim : int
        Input dimension.
    odim : int
        Output dimension.
    dropout_rate : float
        Dropout rate.
    pos_enc : paddle.nn.Layer
        Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(1, odim, 3, 2),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(odim, odim, 3, 2),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(odim, odim, 3, 2),
            paddle.nn.ReLU(), )
        self.out = paddle.nn.Sequential(
            paddle.nn.Linear(odim * (((
                (idim - 1) // 2 - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else
            PositionalEncoding(odim, dropout_rate), )

    def forward(self, x, x_mask):
        """Subsample x.
        Parameters
        ----------
        x : paddle.Tensor
            Input tensor (#batch, time, idim).
        x_mask : paddle.Tensor
            Input mask (#batch, 1, time).
        Returns
        ----------
        paddle.Tensor
            Subsampled tensor (#batch, time', odim),
            where time' = time // 8.
        paddle.Tensor
            Subsampled mask (#batch, 1, time'),
            where time' = time // 8.
        """
        # (b, c, t, f)
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.out(x.transpose([0, 2, 1, 3]).reshape([b, t, c * f]))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
