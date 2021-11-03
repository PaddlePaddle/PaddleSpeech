#!/bin/bash

if [ $# != 1 ];then
    echo "usage: ${0} data_pre_conf"
    echo $1
    exit -1
fi

data_pre_conf=$1

python3 -u ${BIN_DIR}/pre_data.py \
--config ${data_pre_conf}

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi

exit 0
