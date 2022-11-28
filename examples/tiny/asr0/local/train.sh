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

if [ $# -lt 2 ] && [ $# -gt 3 ];then
    echo "usage: CUDA_VISIBLE_DEVICES=0 ${0} config_path ckpt_name ips(optional)"
    exit -1
fi

config_path=$1
ckpt_name=$2
ips=$3

if [ ! $ips ];then
  ips_config=
else
  ips_config="--ips="${ips}
fi

mkdir -p exp

# default memeory allocator strategy may case gpu training hang
# for no OOM raised when memory exhaused
export FLAGS_allocator_strategy=naive_best_fit

if [ ${ngpu} == 0 ]; then
python3 -u ${BIN_DIR}/train.py \
--ngpu ${ngpu} \
--config ${config_path} \
--output exp/${ckpt_name} \
--profiler-options "${profiler_options}" \
--seed ${seed}
else
python3 -m paddle.distributed.launch --gpus=${CUDA_VISIBLE_DEVICES} ${ips_config} ${BIN_DIR}/train.py \
--ngpu ${ngpu} \
--config ${config_path} \
--output exp/${ckpt_name} \
--profiler-options "${profiler_options}" \
--seed ${seed}
fi

if [ ${seed} != 0  ]; then
    unset FLAGS_cudnn_deterministic
fi

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi

exit 0
