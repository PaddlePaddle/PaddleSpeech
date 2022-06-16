#!/bin/bash

set -e

. path.sh

stage=0
stop_stage=100
#tarfile=asr0_deepspeech2_online_wenetspeech_ckpt_1.0.2.model.tar.gz
tarfile=asr0_deepspeech2_online_aishell_fbank161_ckpt_1.0.1.model.tar.gz
model_prefix=avg_1.jit
model=${model_prefix}.pdmodel
param=${model_prefix}.pdiparams

. utils/parse_options.sh

data=data
exp=exp

mkdir -p $data $exp

dir=$data/exp/deepspeech2_online/checkpoints

# wenetspeech or aishell
model_type=$(echo $tarfile | cut -d '_' -f 4)

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
    test -f $data/$tarfile || wget -P $data -c https://paddlespeech.bj.bcebos.com/s2t/$model_type/asr0/$tarfile

    # wenetspeech ds2 model
    pushd $data
    tar zxvf $tarfile 
    popd

    # ds2 model demo inputs
    pushd $exp
    wget -c http://paddlespeech.bj.bcebos.com/speechx/examples/ds2_ol/onnx/static_ds2online_inputs.pickle
    popd
fi

output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    #  prune model by outputs
    mkdir -p $exp/prune

    # prune model deps on output_names.
    ./local/prune.sh $dir $model $param  $output_names  $exp/prune
fi

# aishell rnn hidden is 1024
# wenetspeech rnn hiddn is 2048
if [ $model_type == 'aishell' ];then
    input_shape_dict="{'audio_chunk':[1,-1,161], 'audio_chunk_lens':[1], 'chunk_state_c_box':[5, 1, 1024], 'chunk_state_h_box':[5,1,1024]}"
elif [ $model_type == 'wenetspeech' ];then
    input_shape_dict="{'audio_chunk':[1,-1,161], 'audio_chunk_lens':[1], 'chunk_state_c_box':[5, 1, 2048], 'chunk_state_h_box':[5,1,2048]}"
else
    echo "not support: $model_type"
    exit -1
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    # infer shape by new shape
    mkdir -p $exp/shape
    echo $input_shape_dict
    python3 local/pd_infer_shape.py \
        --model_dir $dir \
        --model_filename $model \
        --params_filename $param \
        --save_dir $exp/shape \
        --input_shape_dict="${input_shape_dict}"
fi

input_file=$exp/static_ds2online_inputs.pickle
test -e $input_file

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
    # to onnx
   ./local/tonnx.sh $dir $model $param $exp/model.onnx

   ./local/infer_check.py --input_file $input_file --model_type $model_type --model_dir $dir --model_prefix $model_prefix --onnx_model $exp/model.onnx
fi


# aishell rnn hidden is 1024
# wenetspeech rnn hiddn is 2048
if [ $model_type == 'aishell' ];then
    input_shape="audio_chunk:1,-1,161  audio_chunk_lens:1 chunk_state_c_box:5,1,1024 chunk_state_h_box:5,1,1024"  
elif [ $model_type == 'wenetspeech' ];then
    input_shape="audio_chunk:1,-1,161  audio_chunk_lens:1 chunk_state_c_box:5,1,2048 chunk_state_h_box:5,1,2048"  
else
    echo "not support: $model_type"
    exit -1
fi
 

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] ;then
    # wenetspeech ds2 model execed 2GB limit, will error.
    # simplifying onnx model
    ./local/onnx_opt.sh $exp/model.onnx $exp/model.opt.onnx  "$input_shape"

    ./local/infer_check.py --input_file $input_file --model_type $model_type  --model_dir $dir --model_prefix $model_prefix --onnx_model $exp/model.opt.onnx
fi
