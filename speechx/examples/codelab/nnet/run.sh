#!/bin/bash
set +x
set -e

. path.sh

# 1. compile
if [ ! -d ${SPEECHX_EXAMPLES} ]; then
    pushd ${SPEECHX_ROOT} 
    bash build.sh
    popd
fi

# 2. download model
if [ ! -f data/model/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz ]; then
    mkdir -p data/model
    pushd data/model
    wget -c https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz
    tar xzfv asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz
    popd
fi

ckpt_dir=./data/model
model_dir=$ckpt_dir/exp/deepspeech2_online/checkpoints/

ds2_model_test_main \
    --model_path=$model_dir/avg_1.jit.pdmodel \
    --param_path=$model_dir/avg_1.jit.pdiparams

