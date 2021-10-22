# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle

__all__ = [
    "id_mask",
    "feature_mask",
    "combine_mask",
    "future_mask",
]


def id_mask(input, padding_index=0, dtype="bool"):
    """Generate mask with input ids. 

    Those positions where the value equals ``padding_index`` correspond to 0 or
    ``False``, otherwise, 1 or ``True``.

    Parameters
    ----------
    input : Tensor [dtype: int]
        The input tensor. It represents the ids.
    padding_index : int, optional
        The id which represents padding, by default 0.
    dtype : str, optional
        Data type of the returned mask, by default "bool".

    Returns
    -------
    Tensor
        The generate mask. It has the same shape as ``input`` does.
    """
    return paddle.cast(input != padding_index, dtype)


def feature_mask(input, axis, dtype="bool"):
    """Compute mask from input features.

    For a input features, represented as batched feature vectors, those vectors
    which all zeros are considerd padding vectors.

    Parameters
    ----------
    input : Tensor [dtype: float]
        The input tensor which represents featues.
    axis : int
        The index of the feature dimension in ``input``. Other dimensions are
        considered ``spatial`` dimensions.
    dtype : str, optional
        Data type of the generated mask, by default "bool"
    Returns
    -------
    Tensor
        The geenrated mask with ``spatial`` shape as mentioned above.

        It has one less dimension than ``input`` does.
    """
    feature_sum = paddle.sum(paddle.abs(input), axis)
    return paddle.cast(feature_sum != 0, dtype)


def combine_mask(mask1, mask2):
    """Combine two mask with multiplication or logical and.

    Parameters
    -----------
    mask1 : Tensor
        The first mask.
    mask2 : Tensor
        The second mask with broadcastable shape with ``mask1``.
    Returns
    --------
    Tensor
        Combined mask.

    Notes
    ------
    It is mainly used to combine the padding mask and no future mask for
    transformer decoder. 

    Padding mask is used to mask padding positions of the decoder inputs and
    no future mask is used to prevent the decoder to see future information.
    """
    if mask1.dtype == paddle.fluid.core.VarDesc.VarType.BOOL:
        return paddle.logical_and(mask1, mask2)
    else:
        return mask1 * mask2


def future_mask(time_steps, dtype="bool"):
    """Generate lower triangular mask.

    It is used at transformer decoder to prevent the decoder to see future
    information.

    Parameters
    ----------
    time_steps : int
        Decoder time steps.
    dtype : str, optional
        The data type of the generate mask, by default "bool".

    Returns
    -------
    Tensor
        The generated mask.
    """
    mask = paddle.tril(paddle.ones([time_steps, time_steps]))
    return paddle.cast(mask, dtype)
