#!/bin/bash

# this script is for memory check, so please run ./run.sh first.

set +x
set -e

. ./path.sh

if [ ! -d ${SPEECHX_TOOLS}/valgrind/install ]; then
  echo "please install valgrind in the speechx tools dir.\n" 
  exit 1
fi

ckpt_dir=./data/model
model_dir=$ckpt_dir/exp/deepspeech2_online/checkpoints/

valgrind --tool=memcheck --track-origins=yes --leak-check=full --show-leak-kinds=all \
  ds2_model_test_main \
  --model_path=$model_dir/avg_1.jit.pdmodel \
  --param_path=$model_dir/avg_1.jit.pdparams
