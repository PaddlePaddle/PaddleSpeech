import os
import re
import csv
import shutil
import urllib.request
import collections.abc
import torch
import tqdm
import pathlib
import paddle
import numpy as np
def batch_pad_right(array: list, mode="constant", value=0):
    """Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Parameters
    ----------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : listf
        List containing proportion for each dimension of original, non-padded values.

    """

    if not len(array):
        raise IndexError("Tensors list must not be empty")

    if len(array) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return np.expand_dims(array[0], 0), np.array([1.0], dtype="float32")
    if not (
        any(
            [array[i].ndim == array[0].ndim for i in range(1, len(array))]
        )
    ):
        raise IndexError("All array must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the first dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(array[0].ndim):
        if dim != 0:
            if not all(
                [x.shape[dim] == array[0].shape[dim] for x in array[1:]]
            ):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for the first one"
                )
        max_shape.append(max([x.shape[dim] for x in array]))

    batched = []
    valid = []
    for t in array:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent[0])

    batched = np.stack(batched)

    return batched, np.array(valid, dtype="float32")

np_str_obj_array_pattern = re.compile(r"[SaUO]")

def pad_right_to(
    array: np.ndarray, target_shape: (list, tuple), mode="constant", value=0,
):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Parameters
    ----------
    tensor : input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == array.ndim
    pads = []  # this contains the abs length of the padding for each dimension.
    valid_vals = []  # this contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= array.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - array.shape[i]])
        valid_vals.append(array.shape[j] / target_shape[j])
        i -= 1
        j += 1
    array = np.pad(array, pads, mode, constant_values=(value, value))

    return array, valid_vals

def mod_default_collate(batch):
    """Makes a tensor from list of batch values.

    Note that this doesn't need to zip(*) values together
    as PaddedBatch connects them already (by key).

    Here the idea is not to error out.

    This is modified from:
    https://github.com/pytorch/pytorch/blob/c0deb231db76dbea8a9d326401417f7d1ce96ed5/torch/utils/data/_utils/collate.py#L42
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, paddle.Tensor):
        out = None
        try:
            if torch.io.get_worker_info() is not None:
                
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        except RuntimeError:  # Unequal size:
            return batch
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        try:
            if (
                elem_type.__name__ == "ndarray"
                or elem_type.__name__ == "memmap"
            ):
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    return batch
                return mod_default_collate([paddle.to_tensor(b, dtype=b.dtype) for b in batch])
            elif elem.shape == ():  # scalars
                return paddle.to_tensor(batch, dtype=batch.dtype)
        except RuntimeError:  # Unequal size
            return batch
    elif isinstance(elem, float):
        return paddle.to_tensor(batch, dtype=paddle.float64)
    elif isinstance(elem, int):
        return paddle.to_tensor(batch, dtype=paddle.int64)
    else:
        return batch