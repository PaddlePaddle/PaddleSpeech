#! /usr/bin/env bash

export FLAGS_sync_nccl_allreduce=0

ngpu=$(echo ${CUDA_VISIBLE_DEVICES} | python -c 'import sys; a = sys.stdin.read(); print(len(a.split(",")));')
echo "using $ngpu gpus..."

python3 -u ${BIN_DIR}/train.py \
--device 'gpu' \
--nproc ${ngpu} \
--config conf/deepspeech2.yaml \
--output ckpt

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi


exit 0
