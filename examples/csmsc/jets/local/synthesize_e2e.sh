#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

stage=0
stop_stage=0


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/synthesize_e2e.py \
        --am=jets_csmsc \
        --config=${config_path} \
        --ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --phones_dict=dump/phone_id_map.txt \
        --output_dir=${train_output_path}/test_e2e \
        --text=${BIN_DIR}/../../assets/sentences.txt \
        --inference_dir=${train_output_path}/inference
fi
