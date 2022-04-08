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
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_stats_file(stats_file):
    values = []
    with open(stats_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            threshold, fa_per_hour, frr = arr
            values.append([float(fa_per_hour), float(frr) * 100])
    values.reverse()
    return np.array(values)


def plot_det_curve(keywords, stats_dir, figure_file, xlim, x_step, ylim,
                   y_step):
    plt.figure(dpi=200)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.size'] = 12

    for index, keyword in enumerate(keywords):
        stats_file = os.path.join(stats_dir, 'stats.' + str(index) + '.txt')
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

    keywords = ['Hey_Snips']
    img_path = os.path.join(os.path.abspath(sys.argv[1]), 'det.png')

    plot_det_curve(keywords,
                   os.path.abspath(sys.argv[1]), img_path, 10, 2, 10, 2)

    print('DET curve image saved to: {}'.format(img_path))
