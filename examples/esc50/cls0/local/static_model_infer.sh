#!/bin/bash

device=$1
model_dir=$2
audio_file=$3

python3 ${BIN_DIR}/deploy/predict.py \
--device ${device} \
--model_dir ${model_dir} \
--wav ${audio_file} 
