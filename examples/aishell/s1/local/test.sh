#! /usr/bin/env bash

if [ $# != 1 ];then
    echo "usage: ${0} ckpt_path"
    exit -1
fi

# download language model
#bash local/download_lm_ch.sh
#if [ $? -ne 0 ]; then
#    exit 1
#fi

python3 -u ${BIN_DIR}/test.py \
--device 'gpu' \
--nproc 1 \
--config conf/conformer.yaml \
--result_file ${1}.rsl \
--checkpoint_path ${1} 

if [ $? -ne 0 ]; then
    echo "Failed in evaluation!"
    exit 1
fi


exit 0
