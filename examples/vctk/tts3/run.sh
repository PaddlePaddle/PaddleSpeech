#!/bin/bash

set -e
source path.sh

gpus=0,1
stage=0
stop_stage=100

conf_path=conf/default.yaml
train_output_path=exp/default
ckpt_name=snapshot_iter_331.pdz

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
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${conf_path} ${train_output_path} || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # synthesize, vocoder is pwgan by default
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # synthesize_e2e, vocoder is pwgan by default
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # inference with static model, vocoder is pwgan by default
    CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh ${train_output_path} || exit -1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # install paddle2onnx
    version=$(echo `pip list |grep "paddle2onnx"` |awk -F" " '{print $2}')
    if [[ -z "$version" || ${version} != '1.0.0' ]]; then
        pip install paddle2onnx==1.0.0
    fi
    ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx fastspeech2_vctk
    # considering the balance between speed and quality, we recommend that you use hifigan as vocoder
    ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx pwgan_vctk
    # ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx hifigan_vctk
    
fi

# inference with onnxruntime, use fastspeech2 + pwgan by default
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    ./local/ort_predict.sh ${train_output_path}
fi

# must run after stage 3 (which stage generated static models)
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    ./local/export2lite.sh ${train_output_path} inference pdlite fastspeech2_vctk x86
    ./local/export2lite.sh ${train_output_path} inference pdlite pwgan_vctk x86
    # ./local/export2lite.sh ${train_output_path} inference pdlite hifigan_vctk x86
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/lite_predict.sh ${train_output_path} || exit -1
fi