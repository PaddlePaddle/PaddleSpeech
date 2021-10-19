#!/bin/bash

python3 ../synthesize.py \
  --config=conf/default.yaml \
  --checkpoint=exp/default/checkpoints/snapshot_iter_35000.pdz_bak\
  --test-metadata=dump/test/norm/metadata.jsonl \
  --output-dir=exp/default/test
