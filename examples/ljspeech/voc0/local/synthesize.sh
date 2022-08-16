#!/bin/bash

input_mel_path=$1
train_output_path=$2
ckpt_name=$3

python ${BIN_DIR}/synthesize.py \
    --input=${input_mel_path} \
    --output=${train_output_path}/wavs/ \
    --checkpoint_path=${train_output_path}/checkpoints/${ckpt_name} \
    --ngpu=1