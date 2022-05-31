#!/bin/bash

# this script is for memory check, so please run ./run.sh first.

set +x
set -e

. ./path.sh

if [ ! -d ${SPEECHX_TOOLS}/valgrind/install ]; then
  echo "please install valgrind in the speechx tools dir.\n" 
  exit 1
fi

model_dir=../paddle_asr_model
feat_wspecifier=./feats.ark
cmvn=./cmvn.ark

valgrind --tool=memcheck --track-origins=yes --leak-check=full --show-leak-kinds=all \
  compute_linear_spectrogram_main \
  --wav_rspecifier=scp:$model_dir/wav.scp \
  --feature_wspecifier=ark,t:$feat_wspecifier \
  --cmvn_write_path=$cmvn

