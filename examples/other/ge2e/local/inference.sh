#!/bin/bash

#generate utterance embedding for each utterance in a dataset.
infer_input=$1
infer_output=$2
train_output_path=$3
ckpt_name=$4

python3 ${BIN_DIR}/inference.py \
    --input=${infer_input} \
    --output=${infer_output} \
    --checkpoint_path=${train_output_path}/checkpoints/${ckpt_name} \
    --ngpu=1

