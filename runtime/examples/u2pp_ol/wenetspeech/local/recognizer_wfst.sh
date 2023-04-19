#!/bin/bash
set -e

data=data
exp=exp
nj=40

. utils/parse_options.sh

mkdir -p $exp
ckpt_dir=./data/model
model_dir=$ckpt_dir/asr1_chunk_conformer_u2pp_wenetspeech_static_1.3.0.model/
aishell_wav_scp=aishell_test.scp
text=$data/test/text

./local/split_data.sh $data $data/$aishell_wav_scp $aishell_wav_scp $nj

lang_dir=./data/lang_test/
graph=$lang_dir/TLG.fst
word_table=$lang_dir/words.txt

if [ ! -f $graph ]; then
    # download ngram, if you want to make graph by yourself, please refer local/run_build_tlg.sh
    mkdir -p $lang_dir
    pushd $lang_dir
    wget -c https://paddlespeech.bj.bcebos.com/speechx/examples/ngram/zh/tlg.zip
    unzip tlg.zip
    popd
fi

utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recognizer_wfst.log \
recognizer_main \
    --use_fbank=true \
    --num_bins=80 \
    --cmvn_file=$model_dir/mean_std.json \
    --model_path=$model_dir/export.jit \
    --graph_path=$lang_dir/TLG.fst \
    --word_symbol_table=$word_table \
    --nnet_decoder_chunk=16 \
    --receptive_field_length=7 \
    --subsampling_rate=4 \
    --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
    --rescoring_weight=0.0 \
    --acoustic_scale=2 \
    --result_wspecifier=ark,t:$data/split${nj}/JOB/result_recognizer_wfst.ark


cat $data/split${nj}/*/result_recognizer_wfst.ark > $exp/aishell_recognizer_wfst
utils/compute-wer.py --char=1 --v=1 $text $exp/aishell_recognizer_wfst > $exp/aishell.recognizer_wfst.err
echo "recognizer test have finished!!!"
echo "please checkout in $exp/aishell.recognizer_wfst.err"
tail -n 7 $exp/aishell.recognizer_wfst.err
