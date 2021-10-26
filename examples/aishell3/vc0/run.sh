#!/bin/bash

set -e
source path.sh

gpus=0
stage=0
stop_stage=100

input=~/datasets/data_aishell3/train
preprocess_path=dump
alignment=./alignment

# not include ".pdparams" here
ge2e_ckpt_path=./ge2e_ckpt_0.3/step-3000000
train_output_path=output
# include ".pdparams" here
ge2e_params_path=${ge2e_ckpt_path}.pdparams
tacotron2_params_path=${train_output_path}/checkpoints/step-1000.pdparams
# pretrained model
# tacotron2_params_path=./tacotron2_aishell3_ckpt_0.3/step-450000.pdparams
waveflow_params_path=./waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams
vc_input=ref_audio
vc_output=syn_audio


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    CUDA_VISIBLE_DEVICES=${gpus} ./local/preprocess.sh ${input} ${preprocess_path} ${alignment} ${ge2e_ckpt_path} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${preprocess_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/voice_cloning.sh ${ge2e_params_path} ${tacotron2_params_path} ${waveflow_params_path} ${vc_input} ${vc_output} || exit -1
fi


