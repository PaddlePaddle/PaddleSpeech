#!/bin/bash
set -e

. path.sh

data=data
exp=exp
nj=20
mkdir -p $exp
ckpt_dir=./data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.3.0.model/
text=$data/test/text

utils/run.pl JOB=1:$nj $data/split${nj}/JOB/decoder.log \
ctc_prefix_beam_search_decoder_main \
    --model_path=$model_dir/export.jit \
    --vocab_path=$model_dir/unit.txt \
    --nnet_decoder_chunk=16 \
    --receptive_field_length=7 \
    --subsampling_rate=4 \
    --feature_rspecifier=scp:$data/split${nj}/JOB/fbank.scp \
    --result_wspecifier=ark,t:$data/split${nj}/JOB/result_decode.ark

cat $data/split${nj}/*/result_decode.ark > $exp/aishell.decode.rsl
utils/compute-wer.py --char=1 --v=1 $text $exp/aishell.decode.rsl > $exp/aishell.decode.err
tail -n 7 $exp/aishell.decode.err