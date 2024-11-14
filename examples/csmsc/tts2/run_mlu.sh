#!/bin/bash

set -e
source path.sh
export CUSTOM_DEVICE_BLACK_LIST=elementwise_max
mlus=0
stage=0
stop_stage=100

conf_path=conf/default.yaml
train_output_path=exp/default
ckpt_name=snapshot_iter_30600.pdz

# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    ./local/preprocess.sh ${conf_path} || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # train model, all `ckpt` under `train_output_path/checkpoints/` dir
    FLAGS_selected_mlus=${mlus} ./local/train_mlu.sh ${conf_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # synthesize, vocoder is pwgan by default
    FLAGS_selected_mlus=${mlus} ./local/synthesize_mlu.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # synthesize_e2e, vocoder is pwgan by default
    FLAGS_selected_mlus=${mlus} ./local/synthesize_e2e_mlu.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # inference with static model
    FLAGS_selected_mlus=${mlus} ./local/inference_mlu.sh ${train_output_path} || exit -1
fi

# paddle2onnx, please make sure the static models are in ${train_output_path}/inference first
# we have only tested the following models so far
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # install paddle2onnx
    pip install paddle2onnx --upgrade
    ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx speedyspeech_csmsc
    # considering the balance between speed and quality, we recommend that you use hifigan as vocoder
    ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx pwgan_csmsc
    # ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx mb_melgan_csmsc
    # ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx hifigan_csmsc
fi

# inference with onnxruntime
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    ./local/ort_predict.sh ${train_output_path}
fi

# must run after stage 3 (which stage generated static models)
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    ./local/export2lite.sh ${train_output_path} inference pdlite speedyspeech_csmsc x86
    ./local/export2lite.sh ${train_output_path} inference pdlite pwgan_csmsc x86
    # ./local/export2lite.sh ${train_output_path} inference pdlite mb_melgan_csmsc x86
    # ./local/export2lite.sh ${train_output_path} inference pdlite hifigan_csmsc x86
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/lite_predict.sh ${train_output_path} || exit -1
fi

# PTQ_static
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/PTQ_static.sh  ${train_output_path} speedyspeech_csmsc || exit -1
fi
