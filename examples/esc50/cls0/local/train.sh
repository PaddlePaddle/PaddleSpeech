#!/bin/bash

ngpu=$1
device=$2
feat_backend=$3

num_epochs=50
batch_size=16
ckpt_dir=./checkpoint
save_freq=10

if [ ${ngpu} -gt 1 ]; then
    python3 -m paddle.distributed.launch --gpus $CUDA_VISIBLE_DEVICES ${BIN_DIR}/train.py \
    --epochs ${num_epochs} \
    --feat_backend ${feat_backend} \
    --batch_size ${batch_size} \
    --checkpoint_dir ${ckpt_dir} \
    --save_freq ${save_freq}
else
    python3 ${BIN_DIR}/train.py \
    --device ${device} \
    --epochs ${num_epochs} \
    --feat_backend ${feat_backend} \
    --batch_size ${batch_size} \
    --checkpoint_dir ${ckpt_dir} \
    --save_freq ${save_freq}
fi
