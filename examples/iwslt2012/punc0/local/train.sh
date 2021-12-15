#!/bin/bash

if [ $# != 3 ];then
    echo "usage: CUDA_VISIBLE_DEVICES=0 ${0} config_path ckpt_name log_dir"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
ckpt_name=$2
log_dir=$3

mkdir -p exp

python3 -u ${BIN_DIR}/train.py \
--ngpu ${ngpu} \
--config ${config_path} \
--output_dir exp/${ckpt_name} \
--log_dir ${log_dir}

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi

exit 0
