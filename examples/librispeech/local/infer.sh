#! /usr/bin/env bash

if [[ $# != 1 ]];
    echo "usage: $0 ckpt-path"
    exit -1
fi

# download language model
bash local/download_lm_en.sh
if [ $? -ne 0 ]; then
    exit 1
fi

python3 -u ${BIN_DIR}/infer.py \
--device 'gpu' \
--nproc 1 \
--config conf/deepspeech2.yaml \
--checkpoint_path ${1} 

if [ $? -ne 0 ]; then
    echo "Failed in inference!"
    exit 1
fi

exit 0
