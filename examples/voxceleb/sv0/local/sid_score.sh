#! /usr/bin/env bash
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


stage=-1

. parse_options.sh || exit -1;

dir=$1
trial=$2

if [ ${stage} -le 0 ]; then
    echo "compute the speaker identification score"
    python3 ./local/sid_score.py \
                --enroll ${dir}/enroll/vector.npz \
                --test ${dir}/test/vector.npz \
                --trial ${trial}

fi

if [ ${stage} -le 1 ]; then
    echo "compute the eer metrics"
    python3 ./local/compute_eer.py ${dir}/model/scores
fi
