#!/bin/bash

python3 inference.py \
  --inference-dir=exp/default/inference \
  --text=../sentences.txt \
  --output-dir=exp/default/pd_infer_out \
  --phones-dict=dump/phone_id_map.txt \
  --tones-dict=dump/tone_id_map.txt
