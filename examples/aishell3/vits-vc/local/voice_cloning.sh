#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3
ge2e_params_path=$4
add_blank=$5
ref_audio_dir=$6
src_audio_path=$7

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/voice_cloning.py \
    --config=${config_path} \
    --ckpt=${train_output_path}/checkpoints/${ckpt_name} \
    --ge2e_params_path=${ge2e_params_path} \
    --phones_dict=dump/phone_id_map.txt \
    --text="凯莫瑞安联合体的经济崩溃迫在眉睫。" \
    --audio-path=${src_audio_path} \
    --input-dir=${ref_audio_dir} \
    --output-dir=${train_output_path}/vc_syn \
    --add-blank=${add_blank}
