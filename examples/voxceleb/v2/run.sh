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
# See ../README.txt for more info about the data, results, algorithms

. ./path.sh
set -e

# set all the available gpu id in a machine
# the first gpu id is 0,
gpus=0,1,2,3

# set the nccl info level, default level is INFO
export NCCL_DEBUG=INFO

stage=5 # start stage, the script start from the stage
stop_stage=50 # stop stage, the script stop at the stop_stage

voxceleb1_root=/home/xiongxinlei/task/PaddleSpeech/examples/voxceleb/v1/demo-10
data=./data/
train_output_path=exp/ecapa_tdnn/
config=./conf/train_ecapa_tdnn.yaml
trial=/home/xiongxinlei/task/PaddleSpeech/examples/voxceleb/v1/demo-10/veri_test.txt
# conf_path=conf/conformer.yaml
# decode_conf_path=conf/tuning/decode.yaml
avg_num=5
# audio_file=data/demo_01_03.wav

. parse_options.sh || exit 1;

avg_ckpt=avg_${avg_num}
# ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}')
# echo "checkpoint name ${ckpt}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare voxceleb1 data, 
    # this script create the data/train and data/dev directory
    local/preprocess.sh --stage 0 --config-conf ${config}\
        ${voxceleb1_root} ${data} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    export CUDA_VISIBLE_DEVICES=${gpus} 
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    ./local/train.sh ${config}  ${train_output_path} || exit -1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    # avg.sh best exp/${ckpt}/checkpoints ${avg_num}
    avg.sh best exp/ecapa_tdnn/model/ ${avg_num}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # test ckpt avg_n
    # export CUDA_VISIBLE_DEVICES=0 
    ./local/generate_enroll_test_data.sh --stage 2 \
                    ${voxceleb1_root} ${trial} ${data} ${config} || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # test ckpt avg_n
    # export CUDA_VISIBLE_DEVICES=0 
    ./local/extract_vector.sh exp/ecapa_tdnn/model/${avg_ckpt} \
                              data/enroll/apply_cmvn_data.feat \
                              ${config} \
                              exp/ecapa_tdnn/model/enroll.vector|| exit -1

    ./local/extract_vector.sh exp/ecapa_tdnn/model/${avg_ckpt} \
                              data/test/apply_cmvn_data.feat \
                              ${config} \
                              exp/ecapa_tdnn/model/test.vector|| exit -1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # ctc alignment of test data
    # export CUDA_VISIBLE_DEVICES=0 
    ./local/sid_score.sh --stage 1 exp/ecapa_tdnn/model/enroll.vector \
                          exp/ecapa_tdnn/model/test.vector \
                          ${trial} || exit -1
fi