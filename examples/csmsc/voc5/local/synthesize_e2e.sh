#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../../synthesize_e2e.py \
    --am=fastspeech2_csmsc \
    --am_config=fastspeech2_nosil_baker_ckpt_0.4/default.yaml \
    --am_ckpt=fastspeech2_nosil_baker_ckpt_0.4/snapshot_iter_76000.pdz \
    --am_stat=fastspeech2_nosil_baker_ckpt_0.4/speech_stats.npy \
    --voc=hifigan_csmsc \
    --voc_config=${config_path} \
    --voc_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
    --voc_stat=dump/train/feats_stats.npy \
    --lang=zh \
    --text=${BIN_DIR}/../../assets/sentences.txt \
    --output_dir=${train_output_path}/test_e2e \
    --phones_dict=dump/phone_id_map.txt \
    --inference_dir=${train_output_path}/inference