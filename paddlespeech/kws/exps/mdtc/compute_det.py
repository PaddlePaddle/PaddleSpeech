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
import json
import os
import sys

from tqdm import tqdm


def load_label_and_score(keyword, label_file, score_file):
    # score_table: {uttid: [keywordlist]}
    score_table = {}
    with open(score_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            key = arr[0]
            current_keyword = arr[1]
            str_list = arr[2:]
            if int(current_keyword) == keyword:
                scores = list(map(float, str_list))
                if key not in score_table:
                    score_table.update({key: scores})
    keyword_table = {}
    filler_table = {}
    filler_duration = 0.0
    with open(label_file, 'r', encoding='utf8') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            assert 'key' in obj
            assert 'txt' in obj
            assert 'duration' in obj
            key = obj['key']
            index = obj['txt']
            duration = obj['duration']
            assert key in score_table
            if index == keyword:
                keyword_table[key] = score_table[key]
            else:
                filler_table[key] = score_table[key]
                filler_duration += duration
    return keyword_table, filler_table, filler_duration


class Args:
    def __init__(self):
        self.test_data = '/ssd3/chenxiaojie06/PaddleSpeech/DeepSpeech/paddlespeech/kws/models/data/test/data.list'
        self.keyword = 0
        self.score_file = os.path.join(
            os.path.abspath(sys.argv[1]), 'score.txt')
        self.stats_file = os.path.join(
            os.path.abspath(sys.argv[1]), 'stats.0.txt')
        self.step = 0.01
        self.window_shift = 50


args = Args()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='compute det curve')
    # parser.add_argument('--test_data', required=True, help='label file')
    # parser.add_argument('--keyword', type=int, default=0, help='keyword label')
    # parser.add_argument('--score_file', required=True, help='score file')
    # parser.add_argument('--step', type=float, default=0.01,
    #                     help='threshold step')
    # parser.add_argument('--window_shift', type=int, default=50,
    #                     help='window_shift is used to skip the frames after triggered')
    # parser.add_argument('--stats_file',
    #                     required=True,
    #                     help='false reject/alarm stats file')
    # args = parser.parse_args()

    window_shift = args.window_shift
    keyword_table, filler_table, filler_duration = load_label_and_score(
        args.keyword, args.test_data, args.score_file)
    print('Filler total duration Hours: {}'.format(filler_duration / 3600.0))
    pbar = tqdm(total=int(1.0 / args.step))
    with open(args.stats_file, 'w', encoding='utf8') as fout:
        keyword_index = int(args.keyword)
        threshold = 0.0
        while threshold <= 1.0:
            num_false_reject = 0
            # transverse the all keyword_table
            for key, score_list in keyword_table.items():
                # computer positive test sample, use the max score of list.
                score = max(score_list)
                if float(score) < threshold:
                    num_false_reject += 1
            num_false_alarm = 0
            # transverse the all filler_table
            for key, score_list in filler_table.items():
                i = 0
                while i < len(score_list):
                    if score_list[i] >= threshold:
                        num_false_alarm += 1
                        i += window_shift
                    else:
                        i += 1
            if len(keyword_table) != 0:
                false_reject_rate = num_false_reject / len(keyword_table)
            num_false_alarm = max(num_false_alarm, 1e-6)
            if filler_duration != 0:
                false_alarm_per_hour = num_false_alarm / \
                    (filler_duration / 3600.0)
            fout.write('{:.6f} {:.6f} {:.6f}\n'.format(
                threshold, false_alarm_per_hour, false_reject_rate))
            threshold += args.step
            pbar.update(1)

    pbar.close()
    print('DET saved to: {}'.format(args.stats_file))
