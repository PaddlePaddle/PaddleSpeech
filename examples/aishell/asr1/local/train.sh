#!/bin/bash

profiler_options=
benchmark_batch_size=0
benchmark_max_step=0

# seed may break model convergence
seed=0

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

if [ ${seed} != 0  ]; then
    export FLAGS_cudnn_deterministic=True
    echo "using seed $seed & FLAGS_cudnn_deterministic=True ..."
fi

if [ $# != 2 ];then
    echo "usage: CUDA_VISIBLE_DEVICES=0 ${0} config_path ckpt_name"
    exit -1
fi

config_path=$1
ckpt_name=$2

mkdir -p exp

python3 -u ${BIN_DIR}/train.py \
--seed ${seed} \
--ngpu ${ngpu} \
--config ${config_path} \
--output exp/${ckpt_name} \
--profiler-options "${profiler_options}" \
--benchmark-batch-size ${benchmark_batch_size} \
--benchmark-max-step ${benchmark_max_step}


if [ ${seed} != 0  ]; then
    unset FLAGS_cudnn_deterministic
fi

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi

exit 0
