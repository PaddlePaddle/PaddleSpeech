#!/bin/bash
set -e
source path.sh

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

stage=$1
stop_stage=100
feat_backend=numpy
audio_file=~/cat.wav
ckpt_dir=./checkpoint/epoch_50
output_dir=./export
infer_device=cpu

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ./local/train.sh ${ngpu} ${feat_backend} || exit -1
    exit 0
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ./local/infer.sh ${audio_file} ${ckpt_dir} ${feat_backend} || exit -1
    exit 0
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ./local/export.sh ${ckpt_dir} ${output_dir} || exit -1
    exit 0
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    ./local/static_model_infer.sh ${infer_device} ${output_dir} ${audio_file} || exit -1
    exit 0
fi
