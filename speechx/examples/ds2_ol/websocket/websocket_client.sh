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

# input
mkdir -p data
data=$PWD/data

# output
aishell_wav_scp=aishell_test.scp
if [ ! -d $data/test ]; then
    pushd $data
    wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_test.zip
    unzip  aishell_test.zip
    popd

    realpath $data/test/*/*.wav > $data/wavlist
    awk -F '/' '{ print $(NF) }' $data/wavlist | awk -F '.' '{ print $1 }' > $data/utt_id
    paste $data/utt_id $data/wavlist > $data/$aishell_wav_scp
fi

export GLOG_logtostderr=1

# websocket client
websocket_client_main \
    --wav_rspecifier=scp:$data/$aishell_wav_scp --streaming_chunk=0.36