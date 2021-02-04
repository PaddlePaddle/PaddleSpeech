#! /usr/bin/env bash

source path.sh

# start demo client
CUDA_VISIBLE_DEVICES=0 \
python3 -u ${MAIN_ROOT}/deploy/demo_client.py \
--host_ip="localhost" \
--host_port=8086 \

if [ $? -ne 0 ]; then
    echo "Failed in starting demo client!"
    exit 1
fi


exit 0
