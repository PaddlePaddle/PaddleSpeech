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

stage=-1
cmvn=true

. parse_options.sh || exit -1;

voxceleb1_root=$1
trial=$2
config_conf=$3
dir=$4

if [ ${stage} -le 0 ]; then
    # this script generate the ./data/enroll and ./data/test directory for speaker identification
    # in speaker identification domain, enroll data is the target spkeaers
    # sometimes, enroll dataset is named train dataset to express the target speakers
    # test data is to identify the specific wav if the target speakers
    echo "Generate the enroll and test data for speaker identification"
    python3 local/generate_enroll_test_data.py \
                                --voxceleb1 ${voxceleb1_root} \
                                --trial ${trial} \
                                --dir ./data
fi

if [ ${stage} -le 1 ]; then
    echo "Compute the feature for enroll and test data"
    for x in enroll test; do
        mkdir -p ${dir}/${x}
        python3 ./local/compute_feature.py --config ${config_conf} \
                                --dataset data/${x}/data.json \
                                --feat data/${x}/feat.npz
    done
fi

if $cmvn; then
    if [ ${stage} -le 2 ]; then
        echo "apply the cmvn to enroll and test data"
        mkdir -p ${dir}/{enroll,test}
        for x in enroll test; do
            
            python3 ./local/apply_cmvn.py \
                    --cmvn ${dir}/cmvn.npz \
                    --feat data/${x}/feat.npz \
                    --target ${dir}/${x}/feat.npz
        done
    fi
fi


