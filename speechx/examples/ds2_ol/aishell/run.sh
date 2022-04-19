#!/bin/bash
set +x
set -e

. path.sh

nj=40


# 1. compile
if [ ! -d ${SPEECHX_EXAMPLES} ]; then
    pushd ${SPEECHX_ROOT} 
    bash build.sh
    popd
fi

# input
mkdir -p data
data=$PWD/data

ckpt_dir=$data/model
model_dir=$ckpt_dir/exp/deepspeech2_online/checkpoints/
vocb_dir=$ckpt_dir/data/lang_char/

# output
mkdir -p exp
exp=$PWD/exp

aishell_wav_scp=aishell_test.scp
if [ ! -d $data/test ]; then
    pushd $data
    wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_test.zip
    unzip  aishell_test.zip
    popd

    realpath $data/test/*/*.wav > $data/wavlist
    awk -F '/' '{ print $(NF) }' $data/wavlist | awk -F '.' '{ print $1 }' > $data/utt_id
    paste $data/utt_id $data/wavlist > $data/$aishell_wav_scp
fi


if [ ! -d $ckpt_dir ]; then
    mkdir -p $ckpt_dir
    wget -P $ckpt_dir -c https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz
    tar xzfv $ckpt_dir/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz -C $ckpt_dir
fi

lm=$data/zh_giga.no_cna_cmn.prune01244.klm
if [ ! -f $lm ]; then
    pushd $data
    wget -c https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
    popd
fi

# 3. make feature
label_file=./aishell_result
wer=./aishell_wer

export GLOG_logtostderr=1

# 3. gen linear feat
cmvn=$PWD/cmvn.ark
cmvn-json2kaldi --json_file=$ckpt_dir/data/mean_std.json --cmvn_write_path=$cmvn


./local/split_data.sh $data $data/$aishell_wav_scp $aishell_wav_scp $nj

utils/run.pl JOB=1:$nj $data/split${nj}/JOB/feat.log \
linear-spectrogram-wo-db-norm-ol \
    --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
    --feature_wspecifier=ark,scp:$data/split${nj}/JOB/feat.ark,$data/split${nj}/JOB/feat.scp \
    --cmvn_file=$cmvn \
    --streaming_chunk=0.36

text=$data/test/text

# 4. recognizer
utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recog.wolm.log \
  ctc-prefix-beam-search-decoder-ol \
    --feature_rspecifier=scp:$data/split${nj}/JOB/feat.scp \
    --model_path=$model_dir/avg_1.jit.pdmodel \
    --params_path=$model_dir/avg_1.jit.pdiparams \
    --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
    --dict_file=$vocb_dir/vocab.txt \
    --result_wspecifier=ark,t:$data/split${nj}/JOB/result

cat $data/split${nj}/*/result > ${label_file}
utils/compute-wer.py --char=1 --v=1 ${label_file} $text > ${wer}

# 4. decode with lm
utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recog.lm.log \
  ctc-prefix-beam-search-decoder-ol \
    --feature_rspecifier=scp:$data/split${nj}/JOB/feat.scp \
    --model_path=$model_dir/avg_1.jit.pdmodel \
    --params_path=$model_dir/avg_1.jit.pdiparams \
    --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
    --dict_file=$vocb_dir/vocab.txt \
    --lm_path=$lm \
    --result_wspecifier=ark,t:$data/split${nj}/JOB/result_lm


cat $data/split${nj}/*/result_lm > ${label_file}_lm
utils/compute-wer.py --char=1 --v=1 ${label_file}_lm $text > ${wer}_lm


graph_dir=./aishell_graph
if [ ! -d $graph_dir ]; then
    wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_graph.zip
    unzip  aishell_graph.zip
fi


# 5. test TLG decoder
utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recog.wfst.log \
  wfst-decoder-ol \
    --feature_rspecifier=scp:$data/split${nj}/JOB/feat.scp \
    --model_path=$model_dir/avg_1.jit.pdmodel \
    --params_path=$model_dir/avg_1.jit.pdiparams \
    --word_symbol_table=$graph_dir/words.txt \
    --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
     --graph_path=$graph_dir/TLG.fst --max_active=7500 \
    --acoustic_scale=1.2 \
    --result_wspecifier=ark,t:$data/split${nj}/JOB/result_tlg


cat $data/split${nj}/*/result_tlg > ${label_file}_tlg
utils/compute-wer.py --char=1 --v=1 ${label_file}_tlg $text > ${wer}_tlg