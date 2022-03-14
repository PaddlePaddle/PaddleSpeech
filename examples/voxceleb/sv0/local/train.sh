#!/bin/bash

dir=$1
exp_dir=$2
conf_path=$3

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo "using $ngpu gpus..."

# train the speaker identification task with voxceleb data
# Note: we will store the log file in exp/log directory
python3 -m paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES \
    ${BIN_DIR}/train.py --device "gpu" --checkpoint-dir ${exp_dir} --augment \
    --data-dir ${dir} --config ${conf_path}


if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi

exit 0