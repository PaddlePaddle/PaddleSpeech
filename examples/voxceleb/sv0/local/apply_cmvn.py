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
apply the utterance cmvn feature

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
        "--feat",
        default="./data/train.feat",
        type=str,
        help="feat")

parser.add_argument(
        "--target",
        default="./data/apply_cmvn_train.feat",
        type=str,
        help="apply cmvn feat")

args = parser.parse_args()
eps = 1e-14
def main():
    logger.info("dataset feats: {}".format(args.feat))
    logger.info("cmvn feats: {}".format(args.cmvn))
    logger.info("target feats: {}".format(args.target))
    paddle.device.set_device("cpu")
    cmvn = paddle.load(args.cmvn)
    mean = cmvn["mean"]
    std = cmvn["std"]
    feats = paddle.load(args.feat)
    utts = feats.keys()
    for utt_id in utts:
        utt_feat = feats[utt_id]["wav"]
        # logger.info("type: {}".format(type(utt_feat)))
        utt_feat = paddle.to_tensor(utt_feat)
        feats[utt_id]["wav"] = paddle.clip((utt_feat - mean) * std, 1e-14).numpy()
    
    paddle.save(feats, args.target)
if __name__ == "__main__":
    main()
