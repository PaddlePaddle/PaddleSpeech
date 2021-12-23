#!/bin/bash

ckpt=$1
output_dir=$2

python3 ${BIN_DIR}/export_model.py \
--checkpoint ${ckpt} \
--output_dir ${output_dir}
