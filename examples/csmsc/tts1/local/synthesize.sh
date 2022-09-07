#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/synthesize.py \
    --transformer-tts-config=${config_path} \
    --transformer-tts-checkpoint=${train_output_path}/checkpoints/${ckpt_name} \
    --transformer-tts-stat=dump/train/speech_stats.npy \
    --waveflow-config=waveflow_ljspeech_ckpt_0.3/config.yaml \
    --waveflow-checkpoint=waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams \
    --test-metadata=dump/test/norm/metadata.jsonl \
    --output-dir=${train_output_path}/test \
    --phones-dict=dump/phone_id_map.txt
