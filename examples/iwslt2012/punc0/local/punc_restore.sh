#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3
text=$4
ckpt_prefix=${ckpt_name%.*}

python3 ${BIN_DIR}/punc_restore.py \
    --config=${config_path} \
    --checkpoint=${train_output_path}/checkpoints/${ckpt_name} \
    --text=${text}
