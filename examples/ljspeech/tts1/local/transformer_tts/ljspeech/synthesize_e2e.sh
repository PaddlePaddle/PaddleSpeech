#!/bin/bash
FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 synthesize_e2e.py \
  --transformer-tts-config=conf/default.yaml \
  --transformer-tts-checkpoint=exp/default/checkpoints/snapshot_iter_201500.pdz \
  --transformer-tts-stat=dump/train/speech_stats.npy \
  --waveflow-config=waveflow_ljspeech_ckpt_0.3/config.yaml \
  --waveflow-checkpoint=waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams \
  --text=../sentences.txt \
  --output-dir=exp/default/test_e2e \
  --device="gpu" \
  --phones-dict=dump/phone_id_map.txt
