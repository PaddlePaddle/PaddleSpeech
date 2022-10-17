#!/bin/bash
set -e

. path.sh

data=data
exp=exp
nj=20
mkdir -p $exp
ckpt_dir=./data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.1.0.model/
aishell_wav_scp=aishell_test.scp

./local/split_data.sh $data $data/$aishell_wav_scp $aishell_wav_scp $nj

utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recognizer.log \
u2_recognizer_main \
    --use_fbank=true \
    --num_bins=80 \
    --cmvn_file=$exp/cmvn.ark \
    --model_path=$model_dir/export.jit \
    --vocab_path=$model_dir/unit.txt \
    --nnet_decoder_chunk=16 \
    --receptive_field_length=7 \
    --subsampling_rate=4 \
    --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
    --result_wspecifier=ark,t:$data/split${nj}/JOB/result_recognizer.ark


cat $data/split${nj}/*/result_recognizer.ark > $exp/${label_file}_recognizer
utils/compute-wer.py --char=1 --v=1 $text $exp/${label_file}_recognizer > $exp/${wer}.recognizer
echo "recognizer test have finished!!!"
echo "please checkout in ${exp}/${wer}.recognizer"
tail -n 7 $exp/${wer}.recognizer