#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3
add_blank=$4

stage=0
stop_stage=0


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/synthesize_e2e.py \
        --config=${config_path} \
        --ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --phones_dict=dump/phone_id_map.txt \
        --speaker_dict=dump/speaker_id_map.txt \
        --spk_id=0 \
        --output_dir=${train_output_path}/test_e2e \
        --text=${BIN_DIR}/../sentences.txt \
        --add-blank=${add_blank}
fi
