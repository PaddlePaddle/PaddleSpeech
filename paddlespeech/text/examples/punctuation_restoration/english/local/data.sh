#!/bin/bash

if [ $# != 1 ];then
    echo "usage: ${0} config_path"
    exit -1
fi

config_path=$1

python3 -u ${BIN_DIR}/pre_data.py \
--config ${config_path}

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi

exit 0
