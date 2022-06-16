#!/bin/bash

if [ $# != 4 ];then
    # local/tonnx.sh data/exp/deepspeech2_online/checkpoints avg_1.jit.pdmodel avg_1.jit.pdiparams exp/model.onnx   
    echo "usage: $0 model_dir model_name param_name onnx_output_name"
    exit 1
fi

dir=$1
model=$2
param=$3
output=$4

pip install paddle2onnx
pip install onnx

# https://github.com/PaddlePaddle/Paddle2ONNX#%E5%91%BD%E4%BB%A4%E8%A1%8C%E8%BD%AC%E6%8D%A2
paddle2onnx --model_dir $dir \
            --model_filename $model \
            --params_filename $param \
            --save_file $output \
            --enable_dev_version True \
            --opset_version 9 \
            --enable_onnx_checker True
            