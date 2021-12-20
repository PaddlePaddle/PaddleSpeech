#!/bin/bash

ge2e_params_path=$1
tacotron2_params_path=$2
waveflow_params_path=$3
vc_input=$4
vc_output=$5

python3 ${BIN_DIR}/voice_cloning.py \
    --ge2e_params_path=${ge2e_params_path} \
    --tacotron2_params_path=${tacotron2_params_path} \
    --waveflow_params_path=${waveflow_params_path} \
    --input-dir=${vc_input} \
    --output-dir=${vc_output}