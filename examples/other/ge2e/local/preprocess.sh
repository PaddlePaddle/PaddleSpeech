#!/bin/bash
datasets_root=$1
preprocess_path=$2
dataset_names=$3

python3 ${BIN_DIR}/preprocess.py \
    --datasets_root=${datasets_root} \
    --output_dir=${preprocess_path} \
    --dataset_names=${dataset_names}