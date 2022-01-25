#!/bin/bash

config_conf=$1
train_output_path=$2

echo "config conf: ${config_conf}"
echo "train output path: ${train_output_path}"

python3 ./local/train.py \
        --config ${config_conf} \
        --output-dir ${train_output_path} \
        --train-metadata ./data/apply_cmvnmanifest_dev.feat \
        --dev-metadata ./data/apply_cmvnmanifest_dev.feat \
        --ngpu 4


# visualdl --logdir ./exp/ecapa_tdnn/visualdl/ --port 8230 --host 10.21.226.177