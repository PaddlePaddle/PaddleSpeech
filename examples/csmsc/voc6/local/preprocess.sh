#!/bin/bash

stage=0
stop_stage=100

config_path=$1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 ${BIN_DIR}/preprocess.py \
        --input=~/datasets/BZNSYP/ \
        --output=dump \
        --dataset=csmsc \
        --config=${config_path} \
        --num-cpu=20
fi
