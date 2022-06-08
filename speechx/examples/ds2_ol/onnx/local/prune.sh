#!/bin/bash 

set -e

if [ $# != 5 ]; then
    echo "usage: $0 model_dir model_filename param_filename outputs_names save_dir"
    exit 1
fi

dir=$1
model=$2
param=$3
outputs=$4
save_dir=$5


python local/pd_prune_model.py \
    --model_dir $dir \
    --model_filename $model \
    --params_filename $param \
    --output_names $outputs \
    --save_dir $save_dir