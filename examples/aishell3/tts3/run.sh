#!/bin/bash

set -e
source path.sh

gpus=0,1
stage=0
stop_stage=100


conf_path=conf/default.yaml
train_output_path=exp/default
ckpt_name=snapshot_iter_482.pdz

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    ./local/preprocess.sh ${conf_path} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `train_output_path/checkpoints/` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # synthesize, vocoder is pwgan
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # synthesize_e2e, vocoder is pwgan
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi
