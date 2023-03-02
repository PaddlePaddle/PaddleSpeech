#!/bin/bash

source_path=$1
output_dir=$2

python3 ${BIN_DIR}/vc.py \
    --source_path=${source_path}\
    --output_dir=${output_dir}