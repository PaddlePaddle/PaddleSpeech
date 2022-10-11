#!/bin/bash
set -x
set -e

. path.sh

# 1. compile
if [ ! -d ${SPEECHX_EXAMPLES} ]; then
    pushd ${SPEECHX_ROOT} 
    bash build.sh
    popd
fi

# 2. download model
if [ ! -f data/model/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model.tar.gz ]; then
    mkdir -p data/model
    pushd data/model
    wget -c https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1/static/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model.tar.gz
    tar xzfv asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model.tar.gz
    popd
fi

# produce wav scp
if [ ! -f data/wav.scp ]; then
    mkdir -p data
    pushd data
    wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav
    echo "utt1 " $PWD/zh.wav > wav.scp
    popd 
fi

data=data
exp=exp
mkdir -p $exp
ckpt_dir=./data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model/


cmvn_json2kaldi_main \
    --json_file  $model_dir/mean_std.json \
    --cmvn_write_path $exp/cmvn.ark \
    --binary=false

echo "convert json cmvn to kaldi ark."

compute_fbank_main \
    --num_bins 80 \
    --wav_rspecifier=scp:$data/wav.scp \
    --cmvn_file=$exp/cmvn.ark \
    --feature_wspecifier=ark,t:$exp/fbank.ark

echo "compute fbank feature."

u2_nnet_main \
    --model_path=$model_dir/export.jit \
    --feature_rspecifier=ark,t:$exp/fbank.ark \
    --nnet_decoder_chunk=16 \
    --receptive_field_length=7 \
    --downsampling_rate=4 \
    --acoustic_scale=1.0 \
    --nnet_encoder_outs_wspecifier=ark,t:$exp/encoder_outs.ark \
    --nnet_prob_wspecifier=ark,t:$exp/logprobs.ark

echo "u2 nnet decode."
