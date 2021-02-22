#! /usr/bin/env bash

# download language model
cd ${MAIN_ROOT}/models/lm > /dev/null
bash download_lm_ch.sh
if [ $? -ne 0 ]; then
    exit 1
fi
cd - > /dev/null


python3 -u ${MAIN_ROOT}/test.py \
--device 'gpu' \
--nproc 1 \
--config conf/deepspeech2.yaml \
--output ckpt

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
