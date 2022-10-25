#!/bin/bash
set -e

. path.sh

data=data
exp=exp
mkdir -p $exp
ckpt_dir=./data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model/

u2_recognizer_main \
    --use_fbank=true \
    --num_bins=80 \
    --cmvn_file=$exp/cmvn.ark \
    --model_path=$model_dir/export.jit \
    --nnet_decoder_chunk=16 \
    --receptive_field_length=7 \
    --subsampling_rate=4 \
    --vocab_path=$model_dir/unit.txt \
    --wav_rspecifier=scp:$data/wav.scp \
    --result_wspecifier=ark,t:$exp/result.ark
