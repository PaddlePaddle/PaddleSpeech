#!/bin/bash

if [ $# != 4 ];then
    echo "usage: ${0} config_path ckpt_path_prefix model_type audio_file"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
ckpt_prefix=$2
model_type=$3
audio_file=$4

# download language model
bash local/download_lm_en.sh
if [ $? -ne 0 ]; then
   exit 1
fi

python3 -u ${BIN_DIR}/test_hub.py \
--ngpu ${ngpu} \
--config ${config_path} \
--result_file ${ckpt_prefix}.rsl \
--checkpoint_path ${ckpt_prefix} \
--model_type ${model_type} \
--audio_file ${audio_file}

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
