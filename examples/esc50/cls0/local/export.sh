#!/bin/bash

ckpt_dir=$1
output_dir=$2

python3 ${BIN_DIR}/export_model.py \
--checkpoint ${ckpt_dir}/model.pdparams \
--output_dir ${output_dir}
