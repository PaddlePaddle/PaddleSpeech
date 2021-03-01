#! /usr/bin/env bash

export FLAGS_sync_nccl_allreduce=0

CUDA_VISIBLE_DEVICES=0 \
python3 -u ${BIN_DIR}/train.py \
--device 'gpu' \
--nproc 1 \
--config conf/deepspeech2.yaml \
--output ckpt

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi


exit 0
