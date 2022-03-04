#!/bin/bash
. ./path.sh
set -e

#######################################################################
# stage 1: train the speaker identification model
# stage 2: test speaker identification 
# stage 3: extract the training embeding to train the LDA and PLDA
######################################################################

# you can set the variable PPAUDIO_HOME to specifiy the downloaded the vox1 and vox2 dataset
# default the dataset is the ~/.paddleaudio/
# export PPAUDIO_HOME=

stage=2
dir=data/                     # data directory
exp_dir=exp/ecapa-tdnn/       # experiment directory
mkdir -p ${dir}

if [ $stage -le 1 ]; then
     # stage 1: train the speaker identification model
     python3 \
          -m paddle.distributed.launch --gpus=0,1,2,3 \
          local/train.py --device "gpu" --checkpoint-dir ${exp_dir} \
          --save-freq 10 --data-dir ${dir} --batch-size 256 --epochs 60
fi

if [ $stage -le 2 ]; then
     # stage 1: train the speaker identification model
     python3 \
          local/speaker_verification_cosine.py \
          --load-checkpoint ${exp_dir}/epoch_40/
fi

