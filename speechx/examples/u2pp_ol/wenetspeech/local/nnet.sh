#!/bin/bash
set -e

. path.sh

nj=20
data=data
exp=exp

mkdir -p $exp
ckpt_dir=./data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.3.0.model/

utils/run.pl JOB=1:$nj $data/split${nj}/JOB/nnet.log \
u2_nnet_main \
    --model_path=$model_dir/export.jit \
    --vocab_path=$model_dir/unit.txt \
    --feature_rspecifier=ark,t:${data}/split${nj}/JOB/fbank.ark \
    --nnet_decoder_chunk=16 \
    --receptive_field_length=7 \
    --subsampling_rate=4 \
    --acoustic_scale=1.0 \
    --nnet_encoder_outs_wspecifier=ark,t:$exp/encoder_outs.ark \
    --nnet_prob_wspecifier=ark,t:$exp/logprobs.ark
echo "u2 nnet decode."
