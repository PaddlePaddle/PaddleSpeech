#!/bin/bash
FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ../synthesize.py \
  --speedyspeech-config=conf/default.yaml \
  --speedyspeech-checkpoint=exp/default/checkpoints/snapshot_iter_11400.pdz \
  --speedyspeech-stat=dump/train/feats_stats.npy \
  --pwg-config=pwg_baker_ckpt_0.4/pwg_default.yaml \
  --pwg-checkpoint=pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz \
  --pwg-stat=pwg_baker_ckpt_0.4/pwg_stats.npy \
  --test-metadata=dump/test/norm/metadata.jsonl \
  --output-dir=exp/default/test \
  --inference-dir=exp/default/inference \
  --phones-dict=dump/phone_id_map.txt \
  --tones-dict=dump/tone_id_map.txt \
  --device="gpu"
