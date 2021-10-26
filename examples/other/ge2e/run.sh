#!/bin/bash

set -e
source path.sh

gpus=0
stage=0
stop_stage=100

datasets_root=~/datasets/GE2E
preprocess_path=dump
dataset_names=librispeech_other
train_output_path=output
infer_input=infer_input
infer_output=infer_output
ckpt_name=step-10000


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    ./local/preprocess.sh ${datasets_root} ${preprocess_path} ${dataset_names} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${preprocess_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh ${infer_input} ${infer_output} ${train_output_path} ${ckpt_name} || exit -1
fi
