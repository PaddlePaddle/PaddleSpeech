#!/bin/bash

set -e
source path.sh

gpus=0,1,2,3
stage=0
stop_stage=100

conf_path=conf/default.yaml
train_output_path=exp/default
ckpt_name=snapshot_iter_153.pdz
add_blank=true

# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    ./local/preprocess.sh ${conf_path} ${add_blank}|| exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `train_output_path/checkpoints/` dir
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # synthesize_e2e, vocoder is pwgan
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name} ${add_blank}|| exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh ${train_output_path} ${add_blank}|| exit -1
fi

# # not ready yet for operator missing in Paddle2ONNX
# # paddle2onnx, please make sure the static models are in ${train_output_path}/inference first
# # we have only tested the following models so far
# if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
#     # install paddle2onnx
#     version=$(echo `pip list |grep "paddle2onnx"` |awk -F" " '{print $2}')
#     if [[ -z "$version" || ${version} != '1.0.0' ]]; then
#         pip install paddle2onnx==1.0.0
#     fi
#     ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx vits_csmsc
# fi

# # inference with onnxruntime
# if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
#     ./local/ort_predict.sh ${train_output_path}
# fi

# # not ready yet for operator missing in Paddle-Lite
# # must run after stage 3 (which stage generated static models)
# if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
#     # NOTE by yuantian 2022.11.21: please compile develop version of Paddle-Lite to export and run TTS models,
#     #                   cause TTS models are supported by https://github.com/PaddlePaddle/Paddle-Lite/pull/9587 
#     #                   and https://github.com/PaddlePaddle/Paddle-Lite/pull/9706
#     ./local/export2lite.sh ${train_output_path} inference pdlite vits_csmsc x86
# fi

# if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
#     CUDA_VISIBLE_DEVICES=${gpus} ./local/lite_predict.sh ${train_output_path} || exit -1
# fi

