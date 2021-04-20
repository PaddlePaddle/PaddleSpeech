#! /usr/bin/env bash

ngpu=$(echo ${CUDA_VISIBLE_DEVICES} | python -c 'import sys; a = sys.stdin.read(); print(len(a.split(",")));')
echo "using $ngpu gpus..."

python3 -u ${BIN_DIR}/train.py \
--device 'gpu' \
--nproc ${ngpu} \
--config conf/conformer.yaml \
--output ckpt-${1}

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi


exit 0
