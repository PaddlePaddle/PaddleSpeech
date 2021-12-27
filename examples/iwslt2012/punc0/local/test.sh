#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

ckpt_prefix=${ckpt_name%.*}

python3 ${BIN_DIR}/test.py \
    --config=${config_path} \
    --checkpoint=${train_output_path}/checkpoints/${ckpt_name}
