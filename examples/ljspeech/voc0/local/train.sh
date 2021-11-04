#!/bin/bash

preprocess_path=$1
train_output_path=$2

python3 ${BIN_DIR}/train.py \
    --data=${preprocess_path} \
    --output=${train_output_path} \
    --ngpu=1