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
import numpy
import numpy as np
import paddle


def waveform_collate_fn(batch):
    """Wrap the waveform into a batch form

    Args:
        batch (list): the waveform list from the dataloader
                      the item of data include several field
                      feat: the utterance waveform data
                      label: the utterance label encoding data

    Returns:
        dict: the batch data to dataloader
    """
    waveforms = np.stack([item['feat'] for item in batch])
    labels = np.stack([item['label'] for item in batch])

    return {'waveforms': waveforms, 'labels': labels}


def feature_normalize(feats: paddle.Tensor,
                      mean_norm: bool=True,
                      std_norm: bool=True,
                      convert_to_numpy: bool=False):
    """Do one utterance feature normalization

    Args:
        feats (paddle.Tensor): the original utterance feat, such as fbank, mfcc
        mean_norm (bool, optional): mean norm flag. Defaults to True.
        std_norm (bool, optional): std norm flag. Defaults to True.
        convert_to_numpy (bool, optional): convert the paddle.tensor to numpy 
                                           and do feature norm with numpy. Defaults to False.

    Returns:
        paddle.Tensor : the normalized feats
    """
    # Features normalization if needed
    # numpy.mean is a little with paddle.mean about 1e-6
    if convert_to_numpy:
        feats_np = feats.numpy()
        mean = feats_np.mean(axis=-1, keepdims=True) if mean_norm else 0
        std = feats_np.std(axis=-1, keepdims=True) if std_norm else 1
        feats_np = (feats_np - mean) / std
        feats = paddle.to_tensor(feats_np, dtype=feats.dtype)
    else:
        mean = feats.mean(axis=-1, keepdim=True) if mean_norm else 0
        std = feats.std(axis=-1, keepdim=True) if std_norm else 1
        feats = (feats - mean) / std

    return feats


def pad_right_2d(x, target_length, axis=-1, mode='constant', **kwargs):
    x = np.asarray(x)
    assert len(
        x.shape) == 2, f'Only 2D arrays supported, but got shape: {x.shape}'

    w = target_length - x.shape[axis]
    assert w >= 0, f'Target length {target_length} is less than origin length {x.shape[axis]}'

    if axis == 0:
        pad_width = [[0, w], [0, 0]]
    else:
        pad_width = [[0, 0], [0, w]]

    return np.pad(x, pad_width, mode=mode, **kwargs)


def batch_feature_normalize(batch, mean_norm: bool=True, std_norm: bool=True):
    """Do batch utterance features normalization

    Args:
        batch (list): the batch feature from dataloader
        mean_norm (bool, optional): mean normalization flag. Defaults to True.
        std_norm (bool, optional): std normalization flag. Defaults to True.

    Returns:
        dict: the normalized batch features
    """
    ids = [item['utt_id'] for item in batch]
    lengths = np.asarray([item['feat'].shape[1] for item in batch])
    feats = list(
        map(lambda x: pad_right_2d(x, lengths.max()),
            [item['feat'] for item in batch]))
    feats = np.stack(feats)

    # Features normalization if needed
    for i in range(len(feats)):
        feat = feats[i][:, :lengths[i]]  # Excluding pad values.
        mean = feat.mean(axis=-1, keepdims=True) if mean_norm else 0
        std = feat.std(axis=-1, keepdims=True) if std_norm else 1
        feats[i][:, :lengths[i]] = (feat - mean) / std
        assert feats[i][:, lengths[
            i]:].sum() == 0  # Padding valus should all be 0.

    # Converts into ratios.
    # the utterance of the max length doesn't need to padding
    # the remaining utterances need to padding and all of them will be padded to max length
    # we convert the original length of each utterance to the ratio of the max length
    lengths = (lengths / lengths.max()).astype(np.float32)

    return {'ids': ids, 'feats': feats, 'lengths': lengths}


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
    valid_vals = []  # this contains the relative lengths for each dimension.
    i = 0  # iterating over target_shape ndims
    while i < len(target_shape):
        assert (target_shape[i] >= array.shape[i]
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

    if not (any(
        [arrays[i].ndim == arrays[0].ndim for i in range(1, len(arrays))])):
        raise IndexError("All arrays must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the last dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(arrays[0].ndim):
        if dim != (arrays[0].ndim - 1):
            if not all(
                [x.shape[dim] == arrays[0].shape[dim] for x in arrays[1:]]):
                raise EnvironmentError(
                    "arrays should have same dimensions except for last one")
        max_shape.append(max([x.shape[dim] for x in arrays]))

    batched = []
    valid = []
    for t in arrays:
        # for each array we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value)
        batched.append(padded)
        valid.append(valid_percent[-1])

    batched = numpy.stack(batched)

    return batched, numpy.array(valid)
