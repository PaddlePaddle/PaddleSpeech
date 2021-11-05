#!/bin/bash

train_output_path=$1
ckpt_name=$2

python3 ${BIN_DIR}/synthesize.py \
    --config=${train_output_path}/config.yaml \
    --checkpoint_path=${train_output_path}/checkpoints/${ckpt_name} \
    --input=${BIN_DIR}/../sentences_en.txt \
    --output=${train_output_path}/test
    --ngpu=1