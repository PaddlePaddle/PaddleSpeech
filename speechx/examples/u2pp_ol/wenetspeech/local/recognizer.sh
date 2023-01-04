#!/bin/bash
set -e

data=data
exp=exp
nj=20

. utils/parse_options.sh

mkdir -p $exp
ckpt_dir=./data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.3.0.model/
aishell_wav_scp=aishell_test.scp
text=$data/test/text

./local/split_data.sh $data $data/$aishell_wav_scp $aishell_wav_scp $nj

utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recognizer.log \
u2_recognizer_main \
    --use_fbank=true \
    --num_bins=80 \
    --cmvn_file=$model_dir/mean_std.json \
    --model_path=$model_dir/export.jit \
    --vocab_path=$model_dir/unit.txt \
    --nnet_decoder_chunk=16 \
    --receptive_field_length=7 \
    --subsampling_rate=4 \
    --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
    --result_wspecifier=ark,t:$data/split${nj}/JOB/result_recognizer.ark


cat $data/split${nj}/*/result_recognizer.ark > $exp/aishell_recognizer
utils/compute-wer.py --char=1 --v=1 $text $exp/aishell_recognizer > $exp/aishell.recognizer.err
echo "recognizer test have finished!!!"
echo "please checkout in $exp/aishell.recognizer.err"
tail -n 7 $exp/aishell.recognizer.err
