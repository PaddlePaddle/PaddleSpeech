#!/bin/bash

if [ $# != 4 ];then
    echo "usage: ${0} config_path decode_config_path ckpt_path_prefix model_type"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
decode_config_path=$2
ckpt_prefix=$3
model_type=$4

# download language model
bash local/download_lm_en.sh
if [ $? -ne 0 ]; then
   exit 1
fi

python3 -u ${BIN_DIR}/test.py \
--ngpu ${ngpu} \
--config ${config_path} \
--decode_cfg ${decode_config_path} \
--result_file ${ckpt_prefix}.rsl \
--checkpoint_path ${ckpt_prefix} \
--model_type ${model_type}

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
