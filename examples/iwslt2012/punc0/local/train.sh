#!/bin/bash

config_path=$1
train_output_path=$2

python3 ${BIN_DIR}/train.py \
    --config=${config_path} \
    --output-dir=${train_output_path} \
    --ngpu=1
