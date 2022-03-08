#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../synthesize.py \
    --config=${config_path} \
    --checkpoint=${train_output_path}/checkpoints/${ckpt_name} \
    --test-metadata=dump/test/norm/metadata.jsonl \
    --output-dir=${train_output_path}/test \
    --generator-type=hifigan
