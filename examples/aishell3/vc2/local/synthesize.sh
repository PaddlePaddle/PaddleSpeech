#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/../synthesize.py \
    --am=fastspeech2_aishell3 \
    --am_config=${config_path} \
    --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
    --am_stat=dump/train/speech_stats.npy \
    --voc=pwgan_aishell3 \
    --voc_config=pwg_aishell3_ckpt_0.5/default.yaml \
    --voc_ckpt=pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
    --voc_stat=pwg_aishell3_ckpt_0.5/feats_stats.npy \
    --test_metadata=dump/test/norm/metadata.jsonl \
    --output_dir=${train_output_path}/test \
    --phones_dict=dump/phone_id_map.txt \
    --speaker_dict=dump/speaker_id_map.txt \
    --voice-cloning=True
