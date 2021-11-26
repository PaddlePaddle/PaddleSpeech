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
"""Subsampling layer definition."""
import paddle
from paddle import nn

from paddlespeech.t2s.modules.transformer.embedding import PositionalEncoding


class Conv2dSubsampling(nn.Layer):
    """Convolutional 2D subsampling (to 1/4 length).
    Parameters
    ----------
    idim : int
        Input dimension.
    odim : int
        Output dimension.
    dropout_rate : float
        Dropout rate.
    pos_enc : nn.Layer
        Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2D(odim, odim, 3, 2),
            nn.ReLU(), )
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
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
        b, c, t, f = paddle.shape(x)
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
