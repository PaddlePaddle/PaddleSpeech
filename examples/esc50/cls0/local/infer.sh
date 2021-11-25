#!/bin/bash

device=$1
audio_file=$2
ckpt_dir=$3
feat_backend=$4

python3 ${BIN_DIR}/predict.py \
--device ${device} \
--wav ${audio_file} \
--feat_backend ${feat_backend} \
--top_k 10 \
--checkpoint ${ckpt_dir}/model.pdparams