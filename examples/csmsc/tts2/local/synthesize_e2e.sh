#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/synthesize_e2e.py \
  --speedyspeech-config=${config_path} \
  --speedyspeech-checkpoint=${train_output_path}/checkpoints/${ckpt_name} \
  --speedyspeech-stat=dump/train/feats_stats.npy \
  --pwg-config=pwg_baker_ckpt_0.4/pwg_default.yaml \
  --pwg-checkpoint=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
  --pwg-stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
  --text=${BIN_DIR}/../sentences.txt \
  --output-dir=${train_output_path}/test_e2e \
  --inference-dir=${train_output_path}/inference \
  --phones-dict=dump/phone_id_map.txt \
  --tones-dict=dump/tone_id_map.txt
