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
# Modified from wekws(https://github.com/wenet-e2e/wekws)
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--cfg_path", type=str, required=True)
parser.add_argument("--keyword", type=str, required=True)
args = parser.parse_args()
# yapf: enable


def load_stats_file(stats_file):
    values = []
    with open(stats_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            threshold, fa_per_hour, frr = arr
            values.append([float(fa_per_hour), float(frr) * 100])
    values.reverse()
    return np.array(values)


def plot_det_curve(keywords, stats_file, figure_file, xlim, x_step, ylim,
                   y_step):
    plt.figure(dpi=200)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.size'] = 12

    for index, keyword in enumerate(keywords):
        values = load_stats_file(stats_file)
        plt.plot(values[:, 0], values[:, 1], label=keyword)

    plt.xlim([0, xlim])
    plt.ylim([0, ylim])
    plt.xticks(range(0, xlim + x_step, x_step))
    plt.yticks(range(0, ylim + y_step, y_step))
    plt.xlabel('False Alarm Per Hour')
    plt.ylabel('False Rejection Rate (\\%)')
    plt.grid(linestyle='--')
    plt.legend(loc='best', fontsize=16)
    plt.savefig(figure_file)


if __name__ == '__main__':
    args.cfg_path = os.path.abspath(os.path.expanduser(args.cfg_path))
    with open(args.cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    scoring_conf = config['scoring']
    img_file = os.path.abspath(scoring_conf['img_file'])
    stats_file = os.path.abspath(scoring_conf['stats_file'])
    keywords = [args.keyword]
    plot_det_curve(keywords, stats_file, img_file, 10, 2, 10, 2)

    print('DET curve image saved to: {}'.format(img_file))
