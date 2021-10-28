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
import argparse
import os
from typing import List

import numpy as np
import paddle

from paddleaudio.backends import load as load_audio
from paddleaudio.features import melspectrogram
from paddleaudio.models.panns import cnn14
from paddleaudio.utils import logger

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='Select which device to predict, defaults to gpu.')
parser.add_argument('--wav', type=str, required=True, help='Audio file to infer.')
parser.add_argument('--sample_duration', type=float, default=2.0, help='Duration(in seconds) of tagging samples to predict.')
parser.add_argument('--hop_duration', type=float, default=0.3, help='Duration(in seconds) between two samples.')
parser.add_argument('--output_dir', type=str, default='./output_dir', help='Directory to save tagging result.')
args = parser.parse_args()
# yapf: enable


def split(waveform: np.ndarray, win_size: int, hop_size: int):
    """
    Split into N waveforms.
    N is decided by win_size and hop_size.
    """
    assert isinstance(waveform, np.ndarray)
    time = []
    data = []
    for i in range(0, len(waveform), hop_size):
        segment = waveform[i:i + win_size]
        if len(segment) < win_size:
            segment = np.pad(segment, (0, win_size - len(segment)))
        data.append(segment)
        time.append(i / len(waveform))
    return time, data


def batchify(data: List[List[float]],
             sample_rate: int,
             batch_size: int,
             **kwargs):
    """
    Extract features from waveforms and create batches.
    """
    examples = []
    for waveform in data:
        feats = melspectrogram(waveform, sample_rate, **kwargs).transpose()
        examples.append(feats)

    # Seperates data into some batches.
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            yield one_batch
            one_batch = []
    if one_batch:
        yield one_batch


def predict(model, data: List[List[float]], sample_rate: int,
            batch_size: int=1):
    """
    Use pretrained model to make predictions.
    """
    batches = batchify(data, sample_rate, batch_size)
    results = None
    model.eval()
    for batch in batches:
        feats = paddle.to_tensor(batch).unsqueeze(1)  \
            # (batch_size, num_frames, num_melbins) -> (batch_size, 1, num_frames, num_melbins)

        audioset_scores = model(feats)
        if results is None:
            results = audioset_scores.numpy()
        else:
            results = np.concatenate((results, audioset_scores.numpy()))

    return results


if __name__ == '__main__':
    paddle.set_device(args.device)
    model = cnn14(pretrained=True, extract_embedding=False)
    waveform, sr = load_audio(args.wav, sr=None)
    time, data = split(waveform,
                       int(args.sample_duration * sr),
                       int(args.hop_duration * sr))
    results = predict(model, data, sr, batch_size=8)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    time = np.arange(0, 1, int(args.hop_duration * sr) / len(waveform))
    output_file = os.path.join(args.output_dir, f'audioset_tagging_sr_{sr}.npz')
    np.savez(output_file, time=time, scores=results)
    logger.info(f'Saved tagging results to {output_file}')
