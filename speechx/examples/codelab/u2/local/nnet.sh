#!/bin/bash
set -x
set -e

. path.sh

data=data
exp=exp
mkdir -p $exp
ckpt_dir=./data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model/

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

