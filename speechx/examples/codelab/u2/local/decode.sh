#!/bin/bash
set +x
set -e

. path.sh

data=data
exp=exp
mkdir -p $exp
ckpt_dir=$data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model/

ctc_prefix_beam_search_decoder_main \
    --model_path=$model_dir/export.jit \
    --nnet_decoder_chunk=16 \
    --receptive_field_length=7 \
    --subsampling_rate=4 \
    --vocab_path=$model_dir/unit.txt \
    --feature_rspecifier=ark,t:$exp/fbank.ark \
    --result_wspecifier=ark,t:$exp/result.ark

echo "u2 ctc prefix beam search decode."
