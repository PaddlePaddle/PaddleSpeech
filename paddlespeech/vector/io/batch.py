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
import numpy as np
import paddle


def waveform_collate_fn(batch):
    waveforms = np.stack([item['feat'] for item in batch])
    labels = np.stack([item['label'] for item in batch])

    return {'waveforms': waveforms, 'labels': labels}


def feature_normalize(feats: paddle.Tensor,
                      mean_norm: bool=True,
                      std_norm: bool=True,
                      convert_to_numpy: bool=False):
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
    ids = [item['id'] for item in batch]
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