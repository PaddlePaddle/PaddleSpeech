#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

stage=0
stop_stage=0

# pwgan
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize.py \
        --am=fastspeech2_ljspeech \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=pwgan_ljspeech \
        --voc_config=pwg_ljspeech_ckpt_0.5/pwg_default.yaml \
        --voc_ckpt=pwg_ljspeech_ckpt_0.5/pwg_snapshot_iter_400000.pdz  \
        --voc_stat=pwg_ljspeech_ckpt_0.5/pwg_stats.npy \
        --test_metadata=dump/test/norm/metadata.jsonl \
        --output_dir=${train_output_path}/test \
        --phones_dict=dump/phone_id_map.txt
fi

# hifigan
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize.py \
        --am=fastspeech2_ljspeech \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=hifigan_ljspeech \
        --voc_config=hifigan_ljspeech_ckpt_0.2.0/default.yaml \
        --voc_ckpt=hifigan_ljspeech_ckpt_0.2.0/snapshot_iter_2500000.pdz \
        --voc_stat=hifigan_ljspeech_ckpt_0.2.0/feats_stats.npy \
        --test_metadata=dump/test/norm/metadata.jsonl \
        --output_dir=${train_output_path}/test \
        --phones_dict=dump/phone_id_map.txt
fi

