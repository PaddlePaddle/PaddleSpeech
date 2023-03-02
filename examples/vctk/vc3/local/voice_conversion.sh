#!/bin/bash

config_path=$1
source_path=$2
output_dir=$3

python3 ${BIN_DIR}/vc.py \
    --config_path=${config_path} \
    --source_path=${source_path}\
    --output_dir=${output_dir} 