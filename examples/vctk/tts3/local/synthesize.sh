#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ${BIN_DIR}/synthesize.py \
  --fastspeech2-config=${config_path} \
  --fastspeech2-checkpoint=${train_output_path}/checkpoints/${ckpt_name} \
  --fastspeech2-stat=dump/train/speech_stats.npy \
  --pwg-config=pwg_vctk_ckpt_0.5/pwg_default.yaml \
  --pwg-checkpoint=pwg_vctk_ckpt_0.5/pwg_snapshot_iter_1000000.pdz \
  --pwg-stat=pwg_vctk_ckpt_0.5/pwg_stats.npy \
  --test-metadata=dump/test/norm/metadata.jsonl \
  --output-dir=${train_output_path}/test \
  --phones-dict=dump/phone_id_map.txt \
  --speaker-dict=dump/speaker_id_map.txt
