#!/bin/bash
. ./path.sh
set -e
export PPAUDIO_HOME=/home/users/xiongxinlei/exprts/v3
dir=./data/
mkdir -p ${dir}
# you can set the variable DATA_HOME to specifiy the downloaded the vox1 and vox2 dataset
python3 \
     local/train.py \
     --data-dir ${dir}
