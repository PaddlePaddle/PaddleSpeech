#!/bin/bash

set -e

. path.sh

stage=0
stop_stage=100

. utils/parse_options.sh

data=data
exp=exp

mkdir -p $data $exp

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
    test -f $data/asr0_deepspeech2_online_wenetspeech_ckpt_1.0.0a.model.tar.gz || wget -c https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr0/asr0_deepspeech2_online_wenetspeech_ckpt_1.0.0a.model.tar.gz -P $data

    pushd $data
    tar zxvf asr0_deepspeech2_online_wenetspeech_ckpt_1.0.0a.model.tar.gz
    popd
fi


dir=$data/exp/deepspeech2_online/checkpoints
model=avg_1.jit.pdmodel
param=avg_1.jit.pdiparams


output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    #  prune model by outputs
    mkdir -p $exp/prune

    # prune model deps on output_names.
    ./local/prune.sh $dir $model $param  $output_names  $exp/prune
fi

input_shape_dict="{'audio_chunk':[1,-1,161], 'audio_chunk_lens':[1], 'chunk_state_c_box':[5, 1, 1024], 'chunk_state_h_box':[5,1,1024]}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    # infer shape by new shape
    mkdir -p $exp/shape
    python3 local/pd_infer_shape.py \
        --model_dir $dir \
        --model_filename $model \
        --params_filename $param \
        --save_dir $exp/shape \
        --input_shape_dict=${input_shape_dict} 
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
    # to onnx
   ./local/tonnx.sh $dir $model $param $exp/model.onnx

   ./local/infer_check.py --input_file 'static_ds2online_inputs.pickle' --model_dir $dir --onnx_model $exp/model.onnx
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] ;then
    input_shape="audio_chunk:1,-1,161  audio_chunk_lens:1 chunk_state_c_box:5,1,1024 chunk_state_h_box:5,1,1024"   
    # simplifying onnx model
    ./local/onnx_opt.sh $exp/model.onnx $exp/model.opt.onnx  $input_shape    

    ./local/infer_check.py --input_file 'static_ds2online_inputs.pickle' --model_dir $dir --onnx_model $exp/model.opt.onnx
fi