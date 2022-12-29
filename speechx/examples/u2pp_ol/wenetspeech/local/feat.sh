#!/bin/bash
set -e

. path.sh

nj=1
stage=-1
stop_stage=100

. utils/parse_options.sh

data=data
exp=exp
mkdir -p $exp

ckpt_dir=./data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.3.0.model/
aishell_wav_scp=aishell_test.scp


#if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#    cmvn_json2kaldi_main \
#        --json_file  $model_dir/mean_std.json \
#        --cmvn_write_path $exp/cmvn.ark \
#        --binary=false
#    
#    echo "convert json cmvn to kaldi ark."
#fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    utils/run.pl JOB=1:$nj feat.log \
    compute_my_fbank_main \
        --wav_rspecifier=scp:aishell_test.scp \
        --feature_wspecifier=ark,scp:fbank.ark,fbank.scp
    
    echo "compute fbank feature."
fi
