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

stage=-1        # start stage, the script start from the stage
stop_stage=100  # stop stage, the script stop at the stop_stage

# in voxceleb1 dataset, there are three trial file, that is veri_test.txt, veri_test2.txt and list_test_hard.txt
# generally, you can choose one of them to do the experiment
voxceleb1_root=/mnt/dataset/voxceleb1/
trial=/mnt/dataset/voxceleb1/veri_test.txt

# once, you start to a new experiment, you should config a new experiment directory
# refer the experiment directory to train_output_path variable
train_output_path=exp/ecapa_tdnn/

# training configuration, cantaing feature, model, loss, epoch etc.
config=./conf/train_ecapa_tdnn.yaml

# do average to get more stable model
avg_num=5

. parse_options.sh || exit 1;

# get the avgerage model
avg_ckpt=avg_${avg_num}



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare voxceleb1 data, 
    # this script create the data/train and data/dev directory
    local/preprocess.sh --stage 0 --config-conf ${config}\
        ${voxceleb1_root} ${train_output_path} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    # set all the available gpu id in a machine
    # the first gpu id is 0,
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    # set the nccl info level, default level is INFO
    # we think that only training process uses multiple gpus
    export NCCL_DEBUG=INFO
    ./local/train.sh ${config}  ${train_output_path} || exit -1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    # avg.sh best exp/${ckpt}/checkpoints ${avg_num}
    avg.sh best ${train_output_path}/model/ ${avg_num}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # generate the enroll and test data for speaker identification domain
    ./local/generate_enroll_test_data.sh --stage 2 \
                    ${voxceleb1_root} ${trial} ${config} ${train_output_path} || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # test ckpt avg_n
    export CUDA_VISIBLE_DEVICES=0
    # use single gpu to extract the enroll and test vector embedding
    for x in enroll test; do
        ./local/extract_vector.sh ${train_output_path}/model/${avg_ckpt} \
            ${train_output_path}/${x} ${config}  || exit -1
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # Generate the of speaker identification score and compute the eer metrics
    echo "Compute the score and eer"
    ./local/sid_score.sh --stage 0 ${train_output_path} ${trial} || exit -1
fi