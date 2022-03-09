#!/bin/bash

. ./path.sh
set -e

#######################################################################
# stage 0: data prepare, including voxceleb1 download and generate {train,dev,enroll,test}.csv
#          voxceleb2 data is m4a format, so we need user to convert the m4a to wav yourselves as described in Readme.md
# stage 1: train the speaker identification model
# stage 2: test speaker identification 
# stage 3: extract the training embeding to train the LDA and PLDA
######################################################################

# you can set the variable PPAUDIO_HOME to specifiy the downloaded the vox1 and vox2 dataset
# default the dataset is the ~/.paddleaudio/
# export PPAUDIO_HOME=

stage=0
dir=data.bak/                     # data directory
exp_dir=exp/ecapa-tdnn/           # experiment directory
mkdir -p ${dir}
mkdir -p ${exp_dir}

if [ $stage -le 0 ]; then 
     # stage 0: data prepare for vox1 and vox2, vox2 must be converted from m4a to wav
     python3 local/data_prepare.py --data-dir ${dir} --augment
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
