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
"""Pseudo QMF modules."""
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from scipy.signal.windows import kaiser


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.

    Args:
        taps (int): 
            The number of filter taps.
        cutoff_ratio (float): 
            Cut-off frequency ratio.
        beta (float): 
            Beta coefficient for kaiser window.
    Returns:
        ndarray:
            Impluse response of prototype filter (taps + 1,).
        .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
            https://ieeexplore.ieee.org/abstract/document/681427
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."
    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps))
    h_i[taps //
        2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(nn.Layer):
    """PQMF module.
    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.
    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122
    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        """Initilize PQMF module.
        The cutoff_ratio and beta parameters are optimized for #subbands = 4.
        See dicussion in https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.

        Args:
            subbands (int): 
                The number of subbands.
            taps (int): 
                The number of filter taps.
            cutoff_ratio (float): 
                Cut-off frequency ratio.
            beta (float): 
                Beta coefficient for kaiser window.
        """
        super().__init__()

        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = (
                2 * h_proto * np.cos((2 * k + 1) * (np.pi / (2 * subbands)) * (
                    np.arange(taps + 1) - (taps / 2)) + (-1)**k * np.pi / 4))
            h_synthesis[k] = (
                2 * h_proto * np.cos((2 * k + 1) * (np.pi / (2 * subbands)) * (
                    np.arange(taps + 1) - (taps / 2)) - (-1)**k * np.pi / 4))

        # convert to tensor
        self.analysis_filter = paddle.to_tensor(
            h_analysis, dtype="float32").unsqueeze(1)
        self.synthesis_filter = paddle.to_tensor(
            h_synthesis, dtype="float32").unsqueeze(0)

        # filter for downsampling & upsampling
        updown_filter = paddle.zeros(
            (subbands, subbands, subbands), dtype="float32")
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.updown_filter = updown_filter
        self.subbands = subbands
        # keep padding info
        self.pad_fn = nn.Pad1D(taps // 2, mode='constant', value=0.0)

    def analysis(self, x):
        """Analysis with PQMF.
        Args:
            x (Tensor): 
                Input tensor (B, 1, T).
        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).
        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.
        Args:
            x (Tensor): 
                Input tensor (B, subbands, T // subbands).
        Returns:
            Tensor: Output tensor (B, 1, T).
        """
        x = F.conv1d_transpose(
            x, self.updown_filter * self.subbands, stride=self.subbands)

        return F.conv1d(self.pad_fn(x), self.synthesis_filter)

    # when converting dygraph to static graph, can not use self.pqmf.synthesis directly
    def forward(self, x):
        return self.synthesis(x)
