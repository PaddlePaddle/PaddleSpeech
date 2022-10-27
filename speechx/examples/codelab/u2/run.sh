#!/bin/bash
set -x
set -e

. path.sh

# 1. compile
if [ ! -d ${SPEECHX_EXAMPLES} ]; then
    pushd ${SPEECHX_ROOT} 
    bash build.sh
    popd
fi

# 2. download model
if [ ! -f data/model/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model.tar.gz ]; then
    mkdir -p data/model
    pushd data/model
    wget -c https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/static/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model.tar.gz
    tar xzfv asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model.tar.gz
    popd
fi

# produce wav scp
if [ ! -f data/wav.scp ]; then
    mkdir -p data
    pushd data
    wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
    echo "utt1 " $PWD/zh.wav > wav.scp
    popd 
fi

data=data
exp=exp
mkdir -p $exp
ckpt_dir=./data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model/


./local/feat.sh

./local/nnet.sh

./local/decode.sh
