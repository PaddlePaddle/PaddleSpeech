#!/bin/bash

python3 ../train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=conf/default.yaml \
    --output-dir=exp/default \
    --nprocs=2 \
    --phones-dict=dump/phone_id_map.txt \
    --speaker-dict=dump/speaker_id_map.txt
