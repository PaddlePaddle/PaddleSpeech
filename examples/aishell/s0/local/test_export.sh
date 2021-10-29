#!/bin/bash

if [ $# != 3 ];then
    echo "usage: ${0} config_path ckpt_path_prefix model_type"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
jit_model_export_path=$2
model_type=$3

# download language model
bash local/download_lm_ch.sh > dev/null 2>&1
if [ $? -ne 0 ]; then
   exit 1
fi

python3 -u ${BIN_DIR}/test_export.py \
--nproc ${ngpu} \
--config ${config_path} \
--result_file ${jit_model_export_path}.rsl \
--export_path ${jit_model_export_path} \
--model_type ${model_type}

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
