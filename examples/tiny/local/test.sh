#! /usr/bin/env bash

# download language model
bash local/download_lm_en.sh
if [ $? -ne 0 ]; then
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 \
python3 -u ${BIN_DIR}/test.py \
--device 'gpu' \
--nproc 1 \
--config conf/deepspeech2.yaml \
--output ckpt


if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
