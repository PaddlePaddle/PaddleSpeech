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

num_epochs=50
batch_size=16
ckpt_dir=./checkpoint
save_freq=10
feat_backend=numpy

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ${ngpu} -gt 1 ]; then
        python -m paddle.distributed.launch --gpus $CUDA_VISIBLE_DEVICES local/train.py \
        --epochs ${num_epochs} \
        --feat_backend ${feat_backend} \
        --batch_size ${batch_size} \
        --checkpoint_dir ${ckpt_dir} \
        --save_freq ${save_freq}
    else
        python local/train.py \
        --device ${device} \
        --epochs ${num_epochs} \
        --feat_backend ${feat_backend} \
        --batch_size ${batch_size} \
        --checkpoint_dir ${ckpt_dir} \
        --save_freq ${save_freq}
    fi
fi

audio_file=~/cat.wav
ckpt=./checkpoint/epoch_50/model.pdparams
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python local/predict.py \
    --device ${device} \
    --wav ${audio_file} \
    --feat_backend ${feat_backend} \
    --top_k 10 \
    --checkpoint ${ckpt}
fi

exit 0