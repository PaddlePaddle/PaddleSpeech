#!/bin/bash

set -e

if [ $# != 3 ];then
    # ./local/onnx_opt.sh model.old.onnx model.opt.onnx  "audio_chunk:1,-1,161  audio_chunk_lens:1 chunk_state_c_box:5,1,1024 chunk_state_h_box:5,1,1024"                                                 
    echo "usage: $0 onnx.model.in onnx.model.out input_shape "
    exit 1
fi

# onnx optimizer
pip install onnx-simplifier

in=$1
out=$2
input_shape=$3

check_n=3

onnxsim $in $2 $check_n --dynamic-input-shape  --input-shape $input_shape 