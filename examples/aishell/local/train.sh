#! /usr/bin/env bash

# train model
# if you wish to resume from an exists model, uncomment --init_from_pretrained_model
export FLAGS_sync_nccl_allreduce=0

python3 -u ${MAIN_ROOT}/train.py \
--device 'gpu' \
--nproc 4 \
--config conf/deepspeech2.yaml \
--output ckpt-${1}


if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi


exit 0
