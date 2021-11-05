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
from typing import List

import numpy as np

from paddlespeech.s2t.utils.log import Log

__all__ = ["pad_list", "pad_sequence", "feat_type"]

logger = Log(__name__).getlog()


def pad_list(sequences: List[np.ndarray],
             padding_value: float=0.0) -> np.ndarray:
    return pad_sequence(sequences, True, padding_value)


def pad_sequence(sequences: List[np.ndarray],
                 batch_first: bool=True,
                 padding_value: float=0.0) -> np.ndarray:
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> a = np.ones([25, 300])
        >>> b = np.ones([22, 300])
        >>> c = np.ones([15, 300])
        >>> pad_sequence([a, b, c]).shape
        [25, 3, 300]

    Note:
        This function returns a np.ndarray of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[np.ndarray]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        np.ndarray of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        np.ndarray of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = np.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def feat_type(filepath):
    suffix = filepath.split(":")[0].split('.')[-1].lower()
    if suffix == 'ark':
        return 'mat'
    elif suffix == 'scp':
        return 'scp'
    elif suffix == 'npy':
        return 'npy'
    elif suffix == 'npz':
        return 'npz'
    elif suffix in ['wav', 'flac']:
        # PCM16
        return 'sound'
    else:
        raise ValueError(f"Not support filetype: {suffix}")
