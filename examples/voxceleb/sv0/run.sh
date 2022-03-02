#!/bin/bash
. ./path.sh
set -e

dir=./data/
mkdir -p ${dir}
# you can set the variable PPAUDIO_HOME to specifiy the downloaded the vox1 and vox2 dataset
python3 \
     local/train.py \
     --data-dir ${dir}
