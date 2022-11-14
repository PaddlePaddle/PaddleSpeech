#!/bin/bash

set -e
source path.sh

gpus=0,1
stage=0
stop_stage=100

conf_path=conf/cnndecoder.yaml
train_output_path=exp/cnndecoder
ckpt_name=snapshot_iter_153.pdz

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
    # synthesize, vocoder is pwgan
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

# synthesize_e2e non-streaming
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # synthesize_e2e, vocoder is pwgan by default
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

# inference non-streaming
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # inference with static model, vocoder is pwgan by default
    CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh ${train_output_path} || exit -1
fi

# synthesize_e2e streaming
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # synthesize_e2e, vocoder is pwgan by default
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_streaming.sh ${conf_path} ${train_output_path} ${ckpt_name} || exit -1
fi

# inference streaming
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # inference with static model, vocoder is pwgan by default
    CUDA_VISIBLE_DEVICES=${gpus} ./local/inference_streaming.sh ${train_output_path} || exit -1
fi

# paddle2onnx non streaming
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    # install paddle2onnx
    version=$(echo `pip list |grep "paddle2onnx"` |awk -F" " '{print $2}')
    if [[ -z "$version" || ${version} != '1.0.0' ]]; then
        pip install paddle2onnx==1.0.0
    fi
    ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx fastspeech2_csmsc
    # considering the balance between speed and quality, we recommend that you use hifigan as vocoder
    ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx pwgan_csmsc
    # ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx mb_melgan_csmsc
    # ./local/paddle2onnx.sh ${train_output_path} inference inference_onnx hifigan_csmsc
fi

# onnxruntime non streaming
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    ./local/ort_predict.sh ${train_output_path}
fi

# paddle2onnx streaming
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    # install paddle2onnx
    version=$(echo `pip list |grep "paddle2onnx"` |awk -F" " '{print $2}')
    if [[ -z "$version" || ${version} != '1.0.0' ]]; then
        pip install paddle2onnx==1.0.0
    fi
    # streaming acoustic model
    ./local/paddle2onnx.sh ${train_output_path} inference_streaming inference_onnx_streaming fastspeech2_csmsc_am_encoder_infer
    ./local/paddle2onnx.sh ${train_output_path} inference_streaming inference_onnx_streaming fastspeech2_csmsc_am_decoder
    ./local/paddle2onnx.sh ${train_output_path} inference_streaming inference_onnx_streaming fastspeech2_csmsc_am_postnet
    # considering the balance between speed and quality, we recommend that you use hifigan as vocoder
    ./local/paddle2onnx.sh ${train_output_path} inference_streaming inference_onnx_streaming pwgan_csmsc
    # ./local/paddle2onnx.sh ${train_output_path} inference_streaming inference_onnx_streaming mb_melgan_csmsc
    # ./local/paddle2onnx.sh ${train_output_path} inference_streaming inference_onnx_streaming hifigan_csmsc
fi

# onnxruntime streaming
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    ./local/ort_predict_streaming.sh ${train_output_path}
fi

# must run after stage 3 (which stage generated static models)
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    # This model is not supported, because 3 ops are not supported on 'arm'. These unsupported ops are: 'round, set_value, share_data'.
    # This model is not supported, because 4 ops are not supported on 'x86'. These unsupported ops are: 'matmul_v2, round, set_value, share_data'.
    ./local/export2lite.sh ${train_output_path} inference pdlite fastspeech2_csmsc x86
    # x86 ok, arm Segmentation fault
    # ./local/export2lite.sh ${train_output_path} inference pdlite pwgan_csmsc x86
    # x86 ok, arm Segmentation fault
    # ./local/export2lite.sh ${train_output_path} inference pdlite mb_melgan_csmsc x86
    # x86 ok, arm ok
    # ./local/export2lite.sh ${train_output_path} inference pdlite hifigan_csmsc x86
fi

# must run after stage 5 (which stage generated static models)
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    # streaming acoustic model
    # This model is not supported, because 3 ops are not supported on 'arm'. These unsupported ops are: 'round, set_value, share_data'.
    # This model is not supported, because 4 ops are not supported on 'x86'. These unsupported ops are: 'matmul_v2, round, set_value, share_data'.
    # ./local/export2lite.sh ${train_output_path} inference pdlite fastspeech2_csmsc x86
    ./local/export2lite.sh ${train_output_path} inference_streaming pdlite_streaming fastspeech2_csmsc_am_encoder_infer x86
    # x86 ok, arm Segmentation fault
    ./local/export2lite.sh ${train_output_path} inference_streaming pdlite_streaming fastspeech2_csmsc_am_decoder x86
    # x86 ok, arm Segmentation fault
    ./local/export2lite.sh ${train_output_path} inference_streaming pdlite_streaming fastspeech2_csmsc_am_postnet x86
    # x86 ok, arm Segmentation fault
    # ./local/export2lite.sh ${train_output_path} inference_streaming pdlite_streaming pwgan_csmsc x86
    # x86 ok, arm Segmentation fault
    # ./local/export2lite.sh ${train_output_path} inference_streaming pdlite_streaming mb_melgan_csmsc x86
    # x86 ok, arm ok
    # ./local/export2lite.sh ${train_output_path} inference_streaming pdlite_streaming hifigan_csmsc x86
fi
