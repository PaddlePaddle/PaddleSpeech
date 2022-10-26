#!/bin/bash
set -x
set -e

. path.sh

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
