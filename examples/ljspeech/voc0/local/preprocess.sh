#!/bin/bash

preprocess_path=$1

python3 ${BIN_DIR}/preprocess.py \
    --input=~/datasets/LJSpeech-1.1 \
    --output=${preprocess_path}