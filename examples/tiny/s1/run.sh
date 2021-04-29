#!/bin/bash
set -e

source path.sh
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1;

# prepare data
bash ./local/data.sh || exit -1

# train model, all `ckpt` under `exp` dir
CUDA_VISIBLE_DEVICES=0 ./local/train.sh conf/conformer.yaml test || exit -1

# avg 1 best model
./local/avg.sh exp/test/checkpoints 1

# test ckpt 1
CUDA_VISIBLE_DEVICES=0 ./local/test.sh conf/conformer.yaml exp/test/checkpoints/avg_1 || exit -1

# export ckpt 1
CUDA_VISIBLE_DEVICES= ./local/export.sh conf/conformer.yaml exp/test/checkpoints/avg_1 exp/test/checkpoints/avg_1.jit.model
