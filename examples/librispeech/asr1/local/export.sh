#!/bin/bash

if [ $# != 3 ];then
    echo "usage: $0 config_path ckpt_prefix jit_model_path"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
ckpt_path_prefix=$2
jit_model_export_path=$3

python3 -u ${BIN_DIR}/export.py \
--ngpu ${ngpu} \
--config ${config_path} \
--checkpoint_path ${ckpt_path_prefix} \
--export_path ${jit_model_export_path}


if [ $? -ne 0 ]; then
    echo "Failed in export!"
    exit 1
fi


exit 0
