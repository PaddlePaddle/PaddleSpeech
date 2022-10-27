#!/bin/bash
set -e

. path.sh

nj=20
stage=-1
stop_stage=100

. utils/parse_options.sh

data=data
exp=exp
mkdir -p $exp

ckpt_dir=./data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.3.0.model/
aishell_wav_scp=aishell_test.scp


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    cmvn_json2kaldi_main \
        --json_file  $model_dir/mean_std.json \
        --cmvn_write_path $exp/cmvn.ark \
        --binary=false
    
    echo "convert json cmvn to kaldi ark."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ./local/split_data.sh $data $data/$aishell_wav_scp $aishell_wav_scp $nj
    
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/feat.log \
    compute_fbank_main \
        --num_bins 80 \
        --cmvn_file=$exp/cmvn.ark \
        --streaming_chunk=36 \
        --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
        --feature_wspecifier=ark,scp:$data/split${nj}/JOB/fbank.ark,$data/split${nj}/JOB/fbank.scp
    
    echo "compute fbank feature."
fi
