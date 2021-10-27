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
import ast
import os
from typing import Dict

import numpy as np
from paddleaudio.utils import logger

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--tagging_file', type=str, required=True, help='')
parser.add_argument('--top_k', type=int, default=10, help='Get top k predicted results of audioset labels.')
parser.add_argument('--smooth', type=ast.literal_eval, default=True, help='Set "True" to apply posterior smoothing.')
parser.add_argument('--smooth_size', type=int, default=5, help='Window size of posterior smoothing.')
parser.add_argument('--label_file', type=str, default='./assets/audioset_labels.txt', help='File of audioset labels.')
parser.add_argument('--output_dir', type=str, default='./output_dir', help='Directory to save tagging labels.')
args = parser.parse_args()
# yapf: enable


def smooth(results: np.ndarray, win_size: int):
    """
    Execute posterior smoothing in-place.
    """
    for i in range(len(results) - 1, -1, -1):
        if i < win_size - 1:
            left = 0
        else:
            left = i + 1 - win_size
        results[i] = np.sum(results[left:i + 1], axis=0) / (i - left + 1)


def generate_topk_label(k: int, label_map: Dict, result: np.ndarray):
    """
    Return top k result.
    """
    result = np.asarray(result)
    topk_idx = (-result).argsort()[:k]

    ret = ''
    for idx in topk_idx:
        label, score = label_map[idx], result[idx]
        ret += f'{label}: {score}\n'
    return ret


if __name__ == "__main__":
    label_map = {}
    with open(args.label_file, 'r') as f:
        for i, l in enumerate(f.readlines()):
            label_map[i] = l.strip()

    results = np.load(args.tagging_file, allow_pickle=True)
    times, scores = results['time'], results['scores']

    if args.smooth:
        logger.info('Posterior smoothing...')
        smooth(scores, win_size=args.smooth_size)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file = os.path.join(
        args.output_dir,
        os.path.basename(args.tagging_file).split('.')[0] + '.txt')
    with open(output_file, 'w') as f:
        for time, score in zip(times, scores):
            f.write(f'{time}\n')
            f.write(generate_topk_label(args.top_k, label_map, score) + '\n')

    logger.info(f'Saved tagging labels to {output_file}')
