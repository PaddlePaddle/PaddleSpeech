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
          local/train.py --device "gpu" --checkpoint-dir ${exp_dir} --augment \
          --save-freq 10 --data-dir ${dir} --batch-size 64 --epochs 100
fi

if [ $stage -le 2 ]; then
     # stage 1: train the speaker identification model
     # you can set the variable PPAUDIO_HOME to specifiy the downloaded the vox1 and vox2 dataset
     python3 \
          local/speaker_verification_cosine.py\
          --batch-size 4 --data-dir ${dir} --load-checkpoint ${exp_dir}/epoch_10/
fi

if [ $stage -le 3 ]; then
     # stage 1: train the speaker identification model
     # you can set the variable PPAUDIO_HOME to specifiy the downloaded the vox1 and vox2 dataset
     python3 \
          local/extract_speaker_embedding.py\
          --audio-path "demo/csv/00001.wav" --load-checkpoint ${exp_dir}/epoch_60/
fi

# if [ $stage -le 3 ]; then
#      # stage 2: extract the training embeding to train the LDA and PLDA
#      # todo: extract the training embedding
# fi 
