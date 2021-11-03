#!/bin/bash
set -e
source path.sh


##  stage, gpu, data_pre_config, train_config, avg_num
if [ $# -lt 4 ]; then
    echo "usage: bash ./run.sh  stage gpu train_config avg_num data_config" 
    echo "eg: bash ./run.sh  0 0 train_config 1 data_config " 
    exit -1
fi

stage=$1
stop_stage=100
gpus=$2
conf_path=$3
avg_num=$4
avg_ckpt=avg_${avg_num}
ckpt=$(basename ${conf_path} | awk -F'.' '{print $1}')
echo "checkpoint name ${ckpt}"

if [ $stage -le 0 ]; then 
    if [ $# -eq 5 ]; then
        data_pre_conf=$5
        # prepare data
        bash ./local/data.sh ${data_pre_conf} || exit -1
    else
        echo "data_pre_conf is not exist!"
        exit -1
    fi
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `exp` dir
    CUDA_VISIBLE_DEVICES=${gpus} bash ./local/train.sh ${conf_path} ${ckpt}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # avg n best model
    bash ./local/avg.sh exp/${ckpt}/checkpoints ${avg_num}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # test ckpt avg_n
   CUDA_VISIBLE_DEVICES=${gpus} bash ./local/test.sh ${conf_path} exp/${ckpt}/checkpoints/${avg_ckpt} || exit -1
fi
