#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../../dygraph_to_static.py \
    --type=voc \
    --voc=pwgan_opencpop \
    --voc_config=${config_path} \
    --voc_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
    --voc_stat=dump/train/feats_stats.npy \
    --inference_dir=exp/default/inference/
