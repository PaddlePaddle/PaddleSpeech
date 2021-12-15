#!/bin/bash
set -e

if [ $# -ne 4 ]; then
    echo "usage: bash ./run.sh stage gpu train_config avg_num" 
    echo "eg: bash ./run.sh  1 0 train_config 1" 
    exit -1
fi

stage=$1
stop_stage=100
gpus=$2
conf_path=$3
avg_num=$4
avg_ckpt=avg_${avg_num}
ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}')
log_dir=log

source path.sh ${ckpt}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # prepare data
    bash ./local/data.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=${gpus} bash ./local/train.sh ${conf_path} ${ckpt} ${log_dir}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # avg n best model
    bash ./local/avg.sh exp/${ckpt}/checkpoints ${avg_num}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
   # test ckpt avg_n
   CUDA_VISIBLE_DEVICES=${gpus} bash ./local/test.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi
