#!/bin/bash
set -e

. path.sh

nj=40
stage=-1
stop_stage=100

. utils/parse_options.sh

# input
data=data
exp=exp
mkdir -p $exp $data

# 1. compile
if [ ! -d ${SPEECHX_BUILD} ]; then
    pushd ${SPEECHX_ROOT} 
    bash build.sh

    # build for android armv8/armv7
    # bash build_android.sh
    popd
fi

ckpt_dir=$data/silero_vad
wav=$data/silero_vad_sample.wav

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
    ./local/download.sh
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ./local/decode.sh 
fi

