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
import numpy as np
import paddle
import paddle.nn.functional as F
from numba import jit
from paddle import nn

from paddlespeech.t2s.modules.masked_fill import masked_fill


class AlignmentModule(nn.Layer):
    """Alignment Learning Framework proposed for parallel TTS models in:
    https://arxiv.org/abs/2108.10447
    """

    def __init__(self, adim, odim):
        super().__init__()
        self.t_conv1 = nn.Conv1D(adim, adim, kernel_size=3, padding=1)
        self.t_conv2 = nn.Conv1D(adim, adim, kernel_size=1, padding=0)

        self.f_conv1 = nn.Conv1D(odim, adim, kernel_size=3, padding=1)
        self.f_conv2 = nn.Conv1D(adim, adim, kernel_size=3, padding=1)
        self.f_conv3 = nn.Conv1D(adim, adim, kernel_size=1, padding=0)

    def forward(self, text, feats, x_masks=None):
        """
        Args:
            text (Tensor): Batched text embedding (B, T_text, adim)
            feats (Tensor): Batched acoustic feature (B, T_feats, odim)
            x_masks (Tensor): Mask tensor (B, T_text)

        Returns:
            Tensor: log probability of attention matrix (B, T_feats, T_text)
        """

        text = text.transpose((0, 2, 1))
        text = F.relu(self.t_conv1(text))
        text = self.t_conv2(text)
        text = text.transpose((0, 2, 1))

        feats = feats.transpose((0, 2, 1))
        feats = F.relu(self.f_conv1(feats))
        feats = F.relu(self.f_conv2(feats))
        feats = self.f_conv3(feats)
        feats = feats.transpose((0, 2, 1))

        dist = feats.unsqueeze(2) - text.unsqueeze(1)
        dist = paddle.linalg.norm(dist, p=2, axis=3)
        score = -dist

        if x_masks is not None:
            x_masks = x_masks.unsqueeze(-2)
            score = masked_fill(score, x_masks, -np.inf)
        log_p_attn = F.log_softmax(score, axis=-1)
        return log_p_attn, score


@jit(nopython=True)
def _monotonic_alignment_search(log_p_attn):
    # https://arxiv.org/abs/2005.11129
    T_mel = log_p_attn.shape[0]
    T_inp = log_p_attn.shape[1]
    Q = np.full((T_inp, T_mel), fill_value=-np.inf)

    log_prob = log_p_attn.transpose(1, 0)  # -> (T_inp,T_mel)
    # 1.  Q <- init first row for all j
    for j in range(T_mel):
        Q[0, j] = log_prob[0, :j + 1].sum()

    # 2. 
    for j in range(1, T_mel):
        for i in range(1, min(j + 1, T_inp)):
            Q[i, j] = max(Q[i - 1, j - 1], Q[i, j - 1]) + log_prob[i, j]

    # 3.
    A = np.full((T_mel, ), fill_value=T_inp - 1)
    for j in range(T_mel - 2, -1, -1):  # T_mel-2, ..., 0
        # 'i' in {A[j+1]-1, A[j+1]}
        i_a = A[j + 1] - 1
        i_b = A[j + 1]
        if i_b == 0:
            argmax_i = 0
        elif Q[i_a, j] >= Q[i_b, j]:
            argmax_i = i_a
        else:
            argmax_i = i_b
        A[j] = argmax_i
    return A


def viterbi_decode(log_p_attn, text_lengths, feats_lengths):
    """
    Args:
        log_p_attn (Tensor): 
            Batched log probability of attention matrix (B, T_feats, T_text)
        text_lengths (Tensor): 
            Text length tensor (B,)
        feats_legnths (Tensor): 
            Feature length tensor (B,)
    Returns:
        Tensor: 
            Batched token duration extracted from `log_p_attn` (B,T_text)
        Tensor: 
            binarization loss tensor ()
    """
    B = log_p_attn.shape[0]
    T_text = log_p_attn.shape[2]
    device = log_p_attn.place

    bin_loss = 0
    ds = paddle.zeros((B, T_text), dtype="int32")
    for b in range(B):
        cur_log_p_attn = log_p_attn[b, :feats_lengths[b], :text_lengths[b]]
        viterbi = _monotonic_alignment_search(cur_log_p_attn.numpy())
        _ds = np.bincount(viterbi)
        ds[b, :len(_ds)] = paddle.to_tensor(
            _ds, place=device, dtype="int32")  

        t_idx = paddle.arange(feats_lengths[b])
        bin_loss = bin_loss - cur_log_p_attn[t_idx, viterbi].mean()
    bin_loss = bin_loss / B
    return ds, bin_loss


@jit(nopython=True)
def _average_by_duration(ds, xs, text_lengths, feats_lengths):
    B = ds.shape[0]
    # xs_avg = np.zeros_like(ds)
    xs_avg = np.zeros(shape=ds.shape, dtype=np.float32)
    ds = ds.astype(np.int32)
    for b in range(B):
        t_text = text_lengths[b]
        t_feats = feats_lengths[b]
        d = ds[b, :t_text]
        d_cumsum = d.cumsum()
        d_cumsum = [0] + list(d_cumsum)
        x = xs[b, :t_feats]
        for n, (start, end) in enumerate(zip(d_cumsum[:-1], d_cumsum[1:])):
            if len(x[start:end]) != 0:
                xs_avg[b, n] = x[start:end].mean()
            else:
                xs_avg[b, n] = 0
    return xs_avg


def average_by_duration(ds, xs, text_lengths, feats_lengths):
    """
    Args:
        ds (Tensor): 
            Batched token duration (B,T_text)
        xs (Tensor): 
            Batched feature sequences to be averaged (B,T_feats)
        text_lengths (Tensor): 
            Text length tensor (B,)
        feats_lengths (Tensor): 
            Feature length tensor (B,)
    Returns:
        Tensor: Batched feature averaged according to the token duration (B, T_text)
    """
    device = ds.place
    args = [ds, xs, text_lengths, feats_lengths]
    args = [arg.numpy() for arg in args]
    xs_avg = _average_by_duration(*args)
    xs_avg = paddle.to_tensor(xs_avg, place=device)
    return xs_avg
