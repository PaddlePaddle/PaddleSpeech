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
Score the speaker embedding between enroll embedding and test embedding
"""

import os
import paddle
import argparse
import paddle.nn as nn
def main(args):
    print("enroll vector: {}".format(args.enroll))
    print("test vector: {}".format(args.test))
    print("trial vector: {}".format(args.trial))

    score_path = os.path.join(os.path.dirname(args.enroll), "scores")
    cos_sim_func = nn.CosineSimilarity(axis=0)
    print("target score : {}".format(score_path))
    enroll_vector = paddle.load(args.enroll)
    test_vector = paddle.load(args.test)
    with open(args.trial, 'r') as f, \
        open(score_path, 'w') as w:
        for line in f:
            target, enroll_utt, test_utt = line.strip().split()
            if enroll_utt in enroll_vector and test_utt in test_vector:
                score = cos_sim_func(enroll_vector[enroll_utt], test_vector[test_utt])
               
                w.write("{} {} {} {}\n".format(target, enroll_utt, test_utt, score.item()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enroll", default="./exp/enroll.vector", type=str, help="enroll utterance embedding")
    parser.add_argument("--test", default="./exp/test.vector", type=str, help="test utterance embedding")
    parser.add_argument("--trial", default="./data/trial", type=str, help="trial file between enroll and test")

    args = parser.parse_args()
    paddle.device.set_device("cpu")
    main(args)