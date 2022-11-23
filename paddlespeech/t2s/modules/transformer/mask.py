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
"""Mask module."""
import paddle


def subsequent_mask(size, dtype=paddle.bool):
    """Create mask for subsequent steps (size, size).

    Args:
        size (int): 
            size of mask
        dtype (paddle.dtype): 
            result dtype
    Return:
        Tensor:
            >>> subsequent_mask(3)
            [[1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]]
    """
    ret = paddle.ones([size, size], dtype=dtype)
    return paddle.tril(ret)


def target_mask(ys_in_pad, ignore_id, dtype=paddle.bool):
    """Create mask for decoder self-attention.

    Args:
        ys_pad (Tensor): 
            batch of padded target sequences (B, Lmax)
        ignore_id (int): 
            index of padding
        dtype (paddle.dtype): 
            result dtype
    Return: 
        Tensor: (B, Lmax, Lmax)
    """
    ys_mask = ys_in_pad != ignore_id
    m = subsequent_mask(ys_mask.shape[-1]).unsqueeze(0)
    return ys_mask.unsqueeze(-2) & m
