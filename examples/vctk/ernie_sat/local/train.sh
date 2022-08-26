#!/bin/bash

config_path=$1
train_output_path=$2

python3 ${BIN_DIR}/train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=${config_path} \
    --output-dir=${train_output_path} \
    --ngpu=8 \
    --phones-dict=dump/phone_id_map.txt