#!/bin/bash
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

if [ $# != 4 ];then
    echo "usage: ${0} checkpoint score_file stats_file"
    exit -1
fi

cfg_path=$1
ckpt=$2
score_file=$3
stats_file=$4

python3 ${BIN_DIR}/score.py --config ${cfg_path} --ckpt ${ckpt} --score_file ${score_file} || exit -1
python3 ${BIN_DIR}/compute_det.py --config ${cfg_path} --score_file ${score_file} --stats_file ${stats_file} || exit -1
