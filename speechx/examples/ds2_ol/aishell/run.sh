#!/bin/bash
set +x
set -e

. path.sh

nj=40
stage=0
stop_stage=100

. utils/parse_options.sh

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

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
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
        tar xzfv $model_dir/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz -C $ckpt_dir
    fi

    lm=$data/zh_giga.no_cna_cmn.prune01244.klm
    if [ ! -f $lm ]; then
        pushd $data
        wget -c https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
        popd
    fi
fi

# 3. make feature
text=$data/test/text
label_file=./aishell_result
wer=./aishell_wer

export GLOG_logtostderr=1


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    # 3. gen linear feat
    cmvn=$data/cmvn.ark
    cmvn-json2kaldi --json_file=$ckpt_dir/data/mean_std.json --cmvn_write_path=$cmvn

    ./local/split_data.sh $data $data/$aishell_wav_scp $aishell_wav_scp $nj

    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/feat.log \
    linear-spectrogram-wo-db-norm-ol \
        --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
        --feature_wspecifier=ark,scp:$data/split${nj}/JOB/feat.ark,$data/split${nj}/JOB/feat.scp \
        --cmvn_file=$cmvn \
        --streaming_chunk=0.36
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ];then
    #  recognizer
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recog.wolm.log \
    ctc-prefix-beam-search-decoder-ol \
        --feature_rspecifier=scp:$data/split${nj}/JOB/feat.scp \
        --model_path=$model_dir/avg_1.jit.pdmodel \
        --params_path=$model_dir/avg_1.jit.pdiparams \
        --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
        --dict_file=$vocb_dir/vocab.txt \
        --result_wspecifier=ark,t:$data/split${nj}/JOB/result

    cat $data/split${nj}/*/result > $exp/${label_file}
    utils/compute-wer.py --char=1 --v=1 $exp/${label_file} $text > $exp/${wer}
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ];then
    #  decode with lm
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recog.lm.log \
    ctc-prefix-beam-search-decoder-ol \
        --feature_rspecifier=scp:$data/split${nj}/JOB/feat.scp \
        --model_path=$model_dir/avg_1.jit.pdmodel \
        --params_path=$model_dir/avg_1.jit.pdiparams \
        --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
        --dict_file=$vocb_dir/vocab.txt \
        --lm_path=$lm \
        --result_wspecifier=ark,t:$data/split${nj}/JOB/result_lm
 
    cat $data/split${nj}/*/result_lm > $exp/${label_file}_lm
    utils/compute-wer.py --char=1 --v=1 $exp/${label_file}_lm $text > $exp/${wer}_lm
fi


wfst=$data/wfst/
mkdir -p $wfst
if [ ! -f $wfst/aishell_graph.zip ]; then
    pushd $wfst
    wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_graph.zip
    unzip aishell_graph.zip
    popd
fi

graph_dir=$wfst/aishell_graph
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    #  TLG decoder
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

    cat $data/split${nj}/*/result_tlg > $exp/${label_file}_tlg
    utils/compute-wer.py --char=1 --v=1 $exp/${label_file}_tlg $text > $exp/${wer}_tlg
fi
