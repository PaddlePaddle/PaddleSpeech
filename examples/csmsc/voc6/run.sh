#!/bin/bash

set -e
source path.sh

gpus=0,1
stage=0
stop_stage=100

conf_path=conf/default.yaml
train_output_path=exp/default
test_input=dump/mel_test
ckpt_name=snapshot_iter_100000.pdz

source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    ./local/preprocess.sh ${conf_path} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # prepare data
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # copy some test mels from dump
    mkdir -p ${test_input}
    cp -r dump/mel/00995*.npy ${test_input}
    # synthesize
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} ${test_input}|| exit -1
fi
