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

# this script extract the data vector embedding

model=$1
dir=$2
config_conf=$3

echo "model: ${model}"
echo "dir: ${dir}"

python3 ./local/extract_vector.py \
                    --model ${model} \
                    --data ${dir}/feat.npz \
                    --config ${config_conf} \
                    --spker-embedding ${dir}/vector.npz