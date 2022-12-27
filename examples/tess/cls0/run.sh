#!/bin/bash
set -e
source path.sh

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

stage=$1
stop_stage=100

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    cfg_path=$2
    ./local/train.sh ${ngpu} ${cfg_path} || exit -1
    exit 0
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    cfg_path=$2
    ./local/infer.sh ${cfg_path} || exit -1
    exit 0
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ckpt=$2
    output_dir=$3
    ./local/export.sh ${ckpt} ${output_dir} || exit -1
    exit 0
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    infer_device=$2
    graph_dir=$3
    audio_file=$4
    ./local/static_model_infer.sh ${infer_device} ${graph_dir} ${audio_file} || exit -1
    exit 0
fi
