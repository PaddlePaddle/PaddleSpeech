#!/bin/bash

config_path=$1
train_output_path=$2

FLAGS_cudnn_exhaustive_search=true \
FLAGS_conv_workspace_size_limit=4000 \
python ${BIN_DIR}/train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=${config_path} \
    --output-dir=${train_output_path} \
    --ngpu=1
