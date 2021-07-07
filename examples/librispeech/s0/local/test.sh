#!/bin/bash

if [ $# != 2 ];then
    echo "usage: ${0} config_path ckpt_path_prefix"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

device=gpu
if [ ${ngpu} == 0 ];then
    device=cpu
fi
config_path=$1
ckpt_prefix=$2

# download language model
bash local/download_lm_en.sh
if [ $? -ne 0 ]; then
   exit 1
fi

python3 -u ${BIN_DIR}/test.py \
--device ${device} \
--nproc 1 \
--config ${config_path} \
--result_file ${ckpt_prefix}.rsl \
--checkpoint_path ${ckpt_prefix}

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
