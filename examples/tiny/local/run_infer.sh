#! /usr/bin/env bash

# download language model
cd $MAIN_ROOT/models/lm > /dev/null
bash download_lm_en.sh
if [ $? -ne 0 ]; then
    exit 1
fi
cd - > /dev/null

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -u ${MAIN_ROOT}/infer.py \
--device 'gpu' \
--nproc 1 \
--config conf/deepspeech2.yaml \
--output ckpt


if [ $? -ne 0 ]; then
    echo "Failed in inference!"
    exit 1
fi

exit 0
