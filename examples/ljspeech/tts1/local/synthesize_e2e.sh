#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/synthesize_e2e.py \
    --transformer-tts-config=${config_path} \
    --transformer-tts-checkpoint=${train_output_path}/checkpoints/${ckpt_name} \
    --transformer-tts-stat=dump/train/speech_stats.npy \
    --waveflow-config=waveflow_ljspeech_ckpt_0.3/config.yaml \
    --waveflow-checkpoint=waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams \
    --text=${BIN_DIR}/../../assets/sentences_en.txt \
    --output-dir=${train_output_path}/test_e2e \
    --phones-dict=dump/phone_id_map.txt
