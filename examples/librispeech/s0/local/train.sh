#!/bin/bash

if [ $# != 3 ];then
    echo "usage: CUDA_VISIBLE_DEVICES=0 ${0} config_path ckpt_name model_type"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

config_path=$1
ckpt_name=$2
model_type=$3

device=gpu
if [ ${ngpu} == 0 ];then
    device=cpu
fi
echo "using ${device}..."

mkdir -p exp

# seed may break model convergence
seed=0
if [ ${seed} != 0 ]; then
    export FLAGS_cudnn_deterministic=True
fi

python3 -u ${BIN_DIR}/train.py \
--device ${device} \
--nproc ${ngpu} \
--config ${config_path} \
--output exp/${ckpt_name} \
--model_type ${model_type} \
--seed ${seed}

if [ ${seed} != 0 ]; then
    unset FLAGS_cudnn_deterministic
fi

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi

exit 0
