#!/bin/bash

set -e
source path.sh

gpus=0
stage=0
stop_stage=100

preprocess_path=preprocessed_ljspeech
train_output_path=output
ckpt_name=step-35000

# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    ./local/preprocess.sh ${preprocess_path} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `train_output_path/checkpoints/` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${preprocess_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # train model, all `ckpt` under `train_output_path/checkpoints/` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${train_output_path} ${ckpt_name} || exit -1
fi

