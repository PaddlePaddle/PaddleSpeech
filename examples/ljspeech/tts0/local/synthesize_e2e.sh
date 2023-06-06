#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3
# TODO: dygraph to static graph is not good for tacotron2_ljspeech now
FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../synthesize_e2e.py \
    --am=tacotron2_ljspeech \
    --am_config=${config_path} \
    --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
    --am_stat=dump/train/speech_stats.npy \
    --voc=pwgan_ljspeech \
    --voc_config=pwg_ljspeech_ckpt_0.5/pwg_default.yaml \
    --voc_ckpt=pwg_ljspeech_ckpt_0.5/pwg_snapshot_iter_400000.pdz  \
    --voc_stat=pwg_ljspeech_ckpt_0.5/pwg_stats.npy \
    --lang=en \
    --text=${BIN_DIR}/../../assets/sentences_en.txt \
    --output_dir=${train_output_path}/test_e2e \
    --phones_dict=dump/phone_id_map.txt \
    # --inference_dir=${train_output_path}/inference