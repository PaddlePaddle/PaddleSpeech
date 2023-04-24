#!/bin/bash
set -e

data=data
exp=exp
nj=20

. utils/parse_options.sh

mkdir -p $exp
ckpt_dir=./data/model
model_dir=$ckpt_dir/onnx_model/
aishell_wav_scp=aishell_test.scp
text=$data/test/text

./local/split_data.sh $data $data/$aishell_wav_scp $aishell_wav_scp $nj

utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recognizer.fd.log \
recognizer_main \
    --use_fbank=true \
    --num_bins=80 \
    --model_path=$model_dir \
    --word_symbol_table=$model_dir/unit.txt \
    --nnet_decoder_chunk=16 \
    --receptive_field_length=7 \
    --subsampling_rate=4 \
    --with_onnx_model=true \
    --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
    --result_wspecifier=ark,t:$data/split${nj}/JOB/recognizer.fd.rsl.ark


cat $data/split${nj}/*/recognizer.fd.rsl.ark > $exp/aishell.recognizer.fd.rsl
utils/compute-wer.py --char=1 --v=1 $text $exp/aishell.recognizer.fd.rsl > $exp/aishell.recognizer.fd.err
echo "recognizer fd test have finished!!!"
echo "please checkout in $exp/aishell.recognizer.fd.err"
tail -n 7 $exp/aishell.recognizer.fd.err
