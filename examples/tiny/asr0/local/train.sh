#!/bin/bash

profiler_options=

# seed may break model convergence
seed=0

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

if [ ${seed} != 0  ]; then
    export FLAGS_cudnn_deterministic=True
    echo "using seed $seed & FLAGS_cudnn_deterministic=True ..."
fi

if [ $# != 3 ];then
    echo "usage: CUDA_VISIBLE_DEVICES=0 ${0} config_path ckpt_name model_type"
    exit -1
fi

config_path=$1
ckpt_name=$2
model_type=$3

mkdir -p exp

python3 -u ${BIN_DIR}/train.py \
--ngpu ${ngpu} \
--config ${config_path} \
--output exp/${ckpt_name} \
--model_type ${model_type} \
--profiler-options "${profiler_options}" \
--seed ${seed}

if [ ${seed} != 0  ]; then
    unset FLAGS_cudnn_deterministic
fi

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi

exit 0
