#!/bin/bash

train_output_path=$1

python3 ${BIN_DIR}/inference.py \
  --inference-dir=${train_output_path}/inference \
  --text=${BIN_DIR}/../sentences.txt \
  --output-dir=${train_output_path}/pd_infer_out \
  --phones-dict=dump/phone_id_map.txt
