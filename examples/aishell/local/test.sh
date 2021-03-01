#! /usr/bin/env bash

# download language model
bash local/download_lm_ch.sh
if [ $? -ne 0 ]; then
    exit 1
fi

python3 -u ${BIN_DIR}/test.py \
--device 'gpu' \
--nproc 1 \
--config conf/deepspeech2.yaml \
--checkpoint_path ${1} 

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
