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
"""Length regulator related modules."""
import paddle
from paddle import nn


class LengthRegulator(nn.Layer):
    """Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or
    phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.

        Parameters
        ----------
        pad_value : float, optional
            Value used for padding.

        """
        super().__init__()
        self.pad_value = pad_value

    def expand(self, encodings: paddle.Tensor,
               durations: paddle.Tensor) -> paddle.Tensor:
        """
        encodings: (B, T, C)
        durations: (B, T)
        """
        batch_size, t_enc = paddle.shape(durations)
        slens = durations.sum(-1)
        t_dec = slens.max()
        M = paddle.zeros([batch_size, t_dec, t_enc])
        for i in range(batch_size):
            k = 0
            for j in range(t_enc):
                d = durations[i, j]
                if d >= 1:
                    M[i, k:k + d, j] = 1
                k += d
        encodings = paddle.matmul(M, encodings)
        return encodings

    def forward(self, xs, ds, alpha=1.0):
        """Calculate forward propagation.

        Parameters
        ----------
        xs : Tensor
            Batch of sequences of char or phoneme embeddings (B, Tmax, D).
        ds : LongTensor
                Batch of durations of each frame (B, T).
        alpha : float, optional
            Alpha value to control speed of speech.

        Returns
        ----------
        Tensor
            replicated input tensor based on durations (B, T*, D).
        """

        if alpha != 1.0:
            assert alpha > 0
            ds = paddle.round(ds.cast(dtype=paddle.float32) * alpha)
        ds = ds.cast(dtype=paddle.int64)
        return self.expand(xs, ds)
