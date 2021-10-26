#!/bin/bash

set -e
source path.sh

gpus=0,1
stage=0
stop_stage=100

preprocess_path=preprocessed_ljspeech
train_output_path=output
# mel generated by Tacotron2
input_mel_path=../tts0/output/test
ckpt_name=step-10000

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    ./local/preprocess.sh ${preprocess_path} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${preprocess_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${input_mel_path} ${train_output_path} ${ckpt_name} || exit -1
fi
