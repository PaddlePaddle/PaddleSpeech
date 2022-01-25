#!/usr/bin/python3
#! coding:utf-8

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
Compute the utterance cmvn feature according to the dataset.

"""

import argparse
import paddle

from paddlespeech.s2t.utils.log import Log 

logger = Log(__name__).getlog()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
        "--cmvn",
        default="./data/cmvn.feat", 
        type=str, 
        help="cmvn file path")
parser.add_argument(
        "--feat-dim",
        default=80,
        type=int,
        help="feat dim")
parser.add_argument(
        "--dataset",
        default=[],
        nargs='*',
        help="dataset to compute the cmvn"
)

args = parser.parse_args()
eps = 1e-14
     
def main():
    all_mean_stat = paddle.zeros(shape=[args.feat_dim])
    all_var_stat = paddle.zeros(shape=[args.feat_dim])
    all_frame_num = 0
    logger.info(args.dataset)
    for dataset_path in args.dataset:
        mean_state, var_state, frame_num = compute_cmvn_statistics(dataset_path, 
                                            args.feat_dim)
        all_frame_num += frame_num
        all_mean_stat += mean_state
        all_var_stat += var_state
        # logger.info("frame num: {}".format(all_frame_num))

    logger.info("frame num: {}".format(all_frame_num))
    all_mean_stat = all_mean_stat / (1.0 * all_frame_num)
    mean = paddle.clip(all_mean_stat, eps)

    all_var_stat = all_var_stat / (1.0 * all_frame_num) - paddle.square(mean)
    all_var_stat = paddle.clip(all_mean_stat, eps)
    std = 1.0 / paddle.sqrt(all_var_stat)
        # 保存 cmvn 数据
    cmvn_info = {
        "mean": mean,
        "std" : std,
        "frame_num": frame_num,
    } 
    logger.info("all mean stat: {}".format(mean))
    logger.info("all var stat: {}".format(std))
    
    paddle.save(cmvn_info, args.cmvn)

def compute_cmvn_statistics(dataset_path, feat_dim):
    logger.info("dataset name: {}".format(dataset_path))
    audio_dataset = paddle.load(dataset_path)
    frame_num = 0
    mean_stat = paddle.zeros(shape=[feat_dim])
    var_stat = paddle.zeros(shape=[feat_dim])
    for utt in audio_dataset.keys():
        feat = audio_dataset[utt]["wav"]
        # logger.info("feat type: {}".format(feat))
        assert(args.feat_dim == feat.shape[1])
        mean_stat += paddle.sum(paddle.to_tensor(feat), axis=0)
        var_stat = paddle.sum(paddle.square(paddle.to_tensor(feat)), axis=0)
        frame_num += feat.shape[0]  
    
    logger.info("dataset mean stat: {}".format(mean_stat))
    logger.info("dataset var stat: {}".format(var_stat))
    logger.info("dataset frame num: {}".format(frame_num))
    return mean_stat, var_stat, frame_num

if __name__ == "__main__":
    main()