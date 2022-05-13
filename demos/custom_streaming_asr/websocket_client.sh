#!/bin/bash
set +x
set -e

. path.sh
# input
data=$PWD/data

# output
wav_scp=wav.scp

export GLOG_logtostderr=1

# websocket client
websocket_client_main \
    --wav_rspecifier=scp:$data/$wav_scp \
    --streaming_chunk=0.36 \
    --port=8881
