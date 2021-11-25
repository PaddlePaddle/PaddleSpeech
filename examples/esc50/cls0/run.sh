#!/bin/bash
set -e
source path.sh

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
if [ ${ngpu} == 0 ];then
    device=cpu
else
    device=gpu
fi

stage=$1
stop_stage=100
feat_backend=numpy

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ./local/train.sh ${ngpu} ${device} ${feat_backend} || exit -1
fi

audio_file=~/cat.wav
ckpt_dir=./checkpoint/epoch_50
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ./local/infer.sh ${device} ${audio_file} ${ckpt_dir} ${feat_backend} || exit -1
fi


exit 0