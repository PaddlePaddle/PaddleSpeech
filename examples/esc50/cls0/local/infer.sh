#!/bin/bash

audio_file=$1
ckpt_dir=$2
feat_backend=$3

python3 ${BIN_DIR}/predict.py \
--wav ${audio_file} \
--feat_backend ${feat_backend} \
--top_k 10 \
--checkpoint ${ckpt_dir}/model.pdparams
