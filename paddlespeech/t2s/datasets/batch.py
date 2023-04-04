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
"""
Utility functions to create batch for arrays which satisfy some conditions.
Batch functions for text sequences, audio and spectrograms are provided.
"""
import numpy as np

__all__ = [
    "batch_text_id",
    "batch_wav",
    "batch_spec",
    "TextIDBatcher",
    "WavBatcher",
    "SpecBatcher",
]


class TextIDBatcher(object):
    """A wrapper class for `batch_text_id`."""
    def __init__(self, pad_id=0, dtype=np.int64):
        self.pad_id = pad_id
        self.dtype = dtype

    def __call__(self, minibatch):
        out = batch_text_id(minibatch, pad_id=self.pad_id, dtype=self.dtype)
        return out


def batch_text_id(minibatch, pad_id=0, dtype=np.int64):
    """Pad sequences to text_ids to the largest length and batch them.
    
    Args:
        minibatch (List[np.ndarray]): list of rank-1 arrays, shape(T,), dtype np.int64, text_ids.
        pad_id (int, optional): the id which correspond to the special pad token. Defaults to 0.
        dtype (np.dtype, optional): the data dtype of the output. Defaults to np.int64.

    Returns:
        np.ndarray: rank-2 array of text_ids, shape(B, T), B stands for batch_size, T stands for length. The output batch.
    """
    peek_example = minibatch[0]
    assert len(peek_example.shape) == 1, "text example is an 1D tensor"
    # assume (channel, n_samples) or (n_samples, )
    lengths = [example.shape[0] for example in minibatch]
    max_len = np.max(lengths)

    batch = []
    for example in minibatch:
        pad_len = max_len - example.shape[0]
        batch.append(
            np.pad(example, [(0, pad_len)],
                   mode='constant',
                   constant_values=pad_id))

    return np.array(batch, dtype=dtype), np.array(lengths, dtype=np.int64)


class WavBatcher(object):
    """A wrapper class for `batch_wav`."""
    def __init__(self, pad_value=0., dtype=np.float32):
        self.pad_value = pad_value
        self.dtype = dtype

    def __call__(self, minibatch):
        out = batch_wav(minibatch, pad_value=self.pad_value, dtype=self.dtype)
        return out


def batch_wav(minibatch, pad_value=0., dtype=np.float32):
    """pad audios to the largest length and batch them.

    Args:
        minibatch (List[np.ndarray]): list of rank-1 float arrays(mono-channel audio, shape(T,)), dtype float.
        pad_value (float, optional): the pad value. Defaults to 0..
        dtype (np.dtype, optional): the data type of the output. Defaults to np.float32.

    Returns:
        np.ndarray: shape(B, T), the output batch.
    """

    peek_example = minibatch[0]
    assert len(peek_example.shape) == 1, "we only handles mono-channel wav"

    # assume (channel, n_samples) or (n_samples, )
    lengths = [example.shape[-1] for example in minibatch]
    max_len = np.max(lengths)

    batch = []
    for example in minibatch:
        pad_len = max_len - example.shape[-1]
        batch.append(
            np.pad(example, [(0, pad_len)],
                   mode='constant',
                   constant_values=pad_value))
    return np.array(batch, dtype=dtype), np.array(lengths, dtype=np.int64)


class SpecBatcher(object):
    """A wrapper class for `batch_spec`"""
    def __init__(self, pad_value=0., time_major=False, dtype=np.float32):
        self.pad_value = pad_value
        self.dtype = dtype
        self.time_major = time_major

    def __call__(self, minibatch):
        out = batch_spec(minibatch,
                         pad_value=self.pad_value,
                         time_major=self.time_major,
                         dtype=self.dtype)
        return out


def batch_spec(minibatch, pad_value=0., time_major=False, dtype=np.float32):
    """Pad spectra to the largest length and batch them.

    Args:
        minibatch (List[np.ndarray]): list of rank-2 arrays of shape(F, T) for mono-channel spectrograms, or list of rank-3 arrays of shape(C, F, T) for multi-channel spectrograms(F stands for frequency bands.), dtype float.
        pad_value (float, optional): the pad value. Defaults to 0..
        dtype (np.dtype, optional): data type of the output. Defaults to np.float32.

    Returns:
        np.ndarray: a rank-3 array of shape(B, F, T) or (B, T, F).
    """
    # assume (F, T) or (T, F)
    peek_example = minibatch[0]
    assert len(
        peek_example.shape) == 2, "we only handles mono channel spectrogram"

    # assume (F, n_frame) or (n_frame, F)
    time_idx = 0 if time_major else -1
    lengths = [example.shape[time_idx] for example in minibatch]
    max_len = np.max(lengths)

    batch = []
    for example in minibatch:
        pad_len = max_len - example.shape[time_idx]
        if time_major:
            batch.append(
                np.pad(example, [(0, pad_len), (0, 0)],
                       mode='constant',
                       constant_values=pad_value))
        else:
            batch.append(
                np.pad(example, [(0, 0), (0, pad_len)],
                       mode='constant',
                       constant_values=pad_value))
    return np.array(batch, dtype=dtype), np.array(lengths, dtype=np.int64)


def batch_sequences(sequences, axis=0, pad_value=0):
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    dtype = seq.dtype
    pad_value = dtype.type(pad_value)
    seq_lengths = [seq.shape[axis] for seq in sequences]
    max_length = np.max(seq_lengths)

    padded_sequences = []
    for seq, length in zip(sequences, seq_lengths):
        padding = [(0, 0)] * axis + [(0, max_length - length)
                                     ] + [(0, 0)] * (ndim - axis - 1)
        padded_seq = np.pad(seq,
                            padding,
                            mode='constant',
                            constant_values=pad_value)
        padded_sequences.append(padded_seq)
    batch = np.stack(padded_sequences)
    return batch
