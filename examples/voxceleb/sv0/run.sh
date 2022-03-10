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

#######################################################################
# stage 0: data prepare, including voxceleb1 download and generate {train,dev,enroll,test}.csv
#          voxceleb2 data is m4a format, so we need user to convert the m4a to wav yourselves as described in Readme.md
# stage 1: train the speaker identification model
# stage 2: test speaker identification 
# stage 3: extract the training embeding to train the LDA and PLDA
######################################################################

# we can set the variable PPAUDIO_HOME to specifiy the root directory of the downloaded vox1 and vox2 dataset 
# default the dataset will be stored in the ~/.paddleaudio/
# the vox2 dataset is stored in m4a format, we need to convert the audio from m4a to wav yourself
# and put all of them to ${PPAUDIO_HOME}/datasets/vox2
# we will find the wav from ${PPAUDIO_HOME}/datasets/vox1/wav and ${PPAUDIO_HOME}/datasets/vox2/wav
# export PPAUDIO_HOME=

stage=0
# data directory
# if we set the variable ${dir}, we will store the wav info to this directory
# otherwise, we will store the wav info to vox1 and vox2 directory respectively
dir=data/                          
exp_dir=exp/ecapa-tdnn/            # experiment directory

# vox2 wav path, we must convert the m4a format to wav format 
# and store them in the ${PPAUDIO_HOME}/datasets/vox2/wav/ directory
vox2_base_path=${PPAUDIO_HOME}/datasets/vox2/wav/
mkdir -p ${dir}
mkdir -p ${exp_dir}

if [ $stage -le 0 ]; then 
     # stage 0: data prepare for vox1 and vox2, vox2 must be converted from m4a to wav
     python3 local/data_prepare.py \
     --data-dir ${dir} --augment --vox2-base-path ${vox2_base_path}
fi 

if [ $stage -le 1 ]; then
     # stage 1: train the speaker identification model
     python3 \
          -m paddle.distributed.launch --gpus=0,1,2,3 \
          ${BIN_DIR}/train.py --device "gpu" --checkpoint-dir ${exp_dir} --augment \
          --data-dir ${dir} --config conf/ecapa_tdnn.yaml
fi

if [ $stage -le 2 ]; then
     # stage 1: get the speaker verification scores with cosine function
     python3 \
          ${BIN_DIR}/speaker_verification_cosine.py\
          --config conf/ecapa_tdnn.yaml \
          --data-dir ${dir} --load-checkpoint ${exp_dir}/epoch_10/
fi

if [ $stage -le 3 ]; then
     # stage 3: extract the audio embedding
     python3 \
          ${BIN_DIR}/extract_speaker_embedding.py\
          --config conf/ecapa_tdnn.yaml \
          --audio-path "demo/csv/00001.wav" --load-checkpoint ${exp_dir}/epoch_60/
fi

# if [ $stage -le 3 ]; then
#      # stage 2: extract the training embeding to train the LDA and PLDA
#      # todo: extract the training embedding
# fi 
