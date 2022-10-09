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


# export can not using StreamdataDataloader, set use_stream_dta False
# u2: reverse_weight should be 0.0
# u2pp: reverse_weight should be same with config file. e.g. 0.3
python3 -u ${BIN_DIR}/export.py \
--ngpu ${ngpu} \
--config ${config_path} \
--opts use_stream_data False \
--checkpoint_path ${ckpt_path_prefix} \
--export_path ${jit_model_export_path}


if [ $? -ne 0 ]; then
    echo "Failed in export!"
    exit 1
fi


exit 0
