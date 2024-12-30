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
"""Generator module in JETS.

This code is based on https://github.com/imdanboy/jets.

"""
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlespeech.t2s.modules.masked_fill import masked_fill


class GaussianUpsampling(nn.Layer):
    """
    Gaussian upsampling with fixed temperature as in:
    https://arxiv.org/abs/2010.04301
    """

    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta

    def forward(self, hs, ds, h_masks=None, d_masks=None):
        """
        Args:
            hs (Tensor): Batched hidden state to be expanded (B, T_text, adim)
            ds (Tensor): Batched token duration (B, T_text)
            h_masks (Tensor): Mask tensor (B,T_feats)
            d_masks (Tensor): Mask tensor (B,T_text)
        Returns:
            Tensor: Expanded hidden state (B, T_feat, adim)
        """
        B = ds.shape[0]

        if h_masks is None:
            T_feats = paddle.to_tensor(ds.sum(), dtype="int32")
        else:
            T_feats = h_masks.shape[-1]
        t = paddle.to_tensor(
            paddle.arange(0, T_feats).unsqueeze(0).tile([B, 1]),
            dtype="float32")
        if h_masks is not None:
            t = t * paddle.to_tensor(h_masks, dtype="float32")

        ds_cumsum = ds.cumsum(axis=-1)
        ds_half = ds / 2
        c = ds_cumsum.astype(ds_half.dtype) - ds_half
        energy = -1 * self.delta * (t.unsqueeze(-1) - c.unsqueeze(1))**2
        if d_masks is not None:
            d_masks = ~(d_masks.unsqueeze(1))
            d_masks.stop_gradient = True
            d_masks = d_masks.tile([1, T_feats, 1])
            energy = masked_fill(energy, d_masks, -float("inf"))
        p_attn = F.softmax(energy, axis=2)  # (B, T_feats, T_text)
        hs = paddle.matmul(p_attn, hs)
        return hs
