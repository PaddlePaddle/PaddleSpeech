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

. ./path.sh
set -e

stage=0
stop_stage=50

# data directory
# if we set the variable ${dir}, we will store the wav info to this directory
# otherwise, we will store the wav info to vox1 and vox2 directory respectively
# vox2 wav path, we must convert the m4a format to wav format    
dir=data/                                 # data info directory   

exp_dir=exp/ecapa-tdnn-vox12-big/            # experiment directory
conf_path=conf/mdtc.yaml          
gpus=0,1,2,3

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

mkdir -p ${exp_dir}

if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
     # stage 0: data prepare for vox1 and vox2, vox2 must be converted from m4a to wav
     bash ./local/data.sh ${dir} ${conf_path}|| exit -1;
fi

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     CUDA_VISIBLE_DEVICES=${gpus} bash ./local/train.sh ${dir} ${exp_dir} ${conf_path} 
fi

if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     CUDA_VISIBLE_DEVICES=0 bash ./local/test.sh ${dir} ${exp_dir} ${conf_path}
fi
