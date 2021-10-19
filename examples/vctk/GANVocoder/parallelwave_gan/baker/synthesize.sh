#!/bin/bash
FLAGS_allocator_strategy=naive_best_fit \
FLAGS_fraction_of_gpu_memory_to_use=0.01 \
python3 ../synthesize.py \
  --config=conf/default.yaml \
  --checkpoint=exp/default/checkpoints/snapshot_iter_400000.pdz\
  --test-metadata=dump/test/norm/metadata.jsonl \
  --output-dir=exp/default/test
