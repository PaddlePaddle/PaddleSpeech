#!/bin/bash

ngpu=$1
cfg_path=$2

if [ ${ngpu} -gt 0 ]; then
    python3 -m paddle.distributed.launch --gpus $CUDA_VISIBLE_DEVICES ${BIN_DIR}/train.py \
    --cfg_path ${cfg_path}
else
    python3 ${BIN_DIR}/train.py \
    --cfg_path ${cfg_path}
fi
