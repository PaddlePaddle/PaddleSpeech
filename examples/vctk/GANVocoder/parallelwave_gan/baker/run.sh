#!/bin/bash

FLAGS_cudnn_exhaustive_search=true \
FLAGS_conv_workspace_size_limit=4000 \
python ../train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=conf/default.yaml \
    --output-dir=exp/default \
    --nprocs=1
