#! /usr/bin/env bash

cd ../.. > /dev/null

# start demo client
CUDA_VISIBLE_DEVICES=0 \
python -u deploy/demo_client.py \
--host_ip='localhost' \
--host_port=8086 \

if [ $? -ne 0 ]; then
    echo "Failed in starting demo client!"
    exit 1
fi


exit 0
