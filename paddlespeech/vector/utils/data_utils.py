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

"""
data utilities
"""
import os
import sys
import numpy
import paddle


def pad_right_to(array, target_shape, mode="constant", value=0):
    """
    This function takes a numpy array of arbitrary shape and pads it to target
    shape by appending values on the right.

    Args:
        array: input numpy array. Input array whose dimension we need to pad.
    target_shape : (list, tuple). Target shape we want for the target array its len must be equal to array.ndim
    mode : str. Pad mode, please refer to numpy.pad documentation.
    value : float. Pad value, please refer to numpy.pad documentation.

    Returns:
        array: numpy.array. Padded array.
        valid_vals : list. List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == array.ndim
    pads = []  # this contains the abs length of the padding for each dimension.
    valid_vals = []  # thic contains the relative lengths for each dimension.
    i = 0 # iterating over target_shape ndims
    while i < len(target_shape):
        assert (
            target_shape[i] >= array.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.append([0, target_shape[i] - array.shape[i]])
        valid_vals.append(array.shape[i] / target_shape[i])
        i += 1

    array = numpy.pad(array, pads, mode=mode, constant_values=value)

    return array, valid_vals


def batch_pad_right(arrays, mode="constant", value=0):
    """Given a list of numpy arrays it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Args:
        arrays : list. List of array we wish to pad together.
        mode : str. Padding mode see numpy.pad documentation.
        value : float. Padding value see numpy.pad documentation.

    Returns:
        array : numpy.array. Padded array.
        valid_vals : list. List containing proportion for each dimension of original, non-padded values.
    """

    if not len(arrays):
        raise IndexError("arrays list must not be empty")

    if len(arrays) == 1:
        # if there is only one array in the batch we simply unsqueeze it.
        return numpy.expand_dims(arrays[0], axis=0), numpy.array([1.0])

    if not (
        any(
            [arrays[i].ndim == arrays[0].ndim for i in range(1, len(arrays))]
        )
    ):
        raise IndexError("All arrays must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the last dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(arrays[0].ndim):
        if dim != (arrays[0].ndim - 1):
            if not all(
                [x.shape[dim] == arrays[0].shape[dim] for x in arrays[1:]]
            ):
                raise EnvironmentError(
                    "arrays should have same dimensions except for last one"
                )
        max_shape.append(max([x.shape[dim] for x in arrays]))

    batched = []
    valid = []
    for t in arrays:
        # for each array we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent[-1])

    batched = numpy.stack(batched)

    return batched, numpy.array(valid)


def length_to_mask(length, max_len=None, dtype=None):
    """Creates a binary mask for each sequence.
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = paddle.cast(paddle.max(length), dtype="int64")  # using arange to generate mask
    mask = paddle.arange(max_len, dtype=length.dtype).expand([paddle.shape(length)[0], max_len]) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    mask = paddle.cast(mask, dtype=dtype)
    return mask
