#!/bin/bash
set -x
set -e

. path.sh

nj=40
stage=0
stop_stage=100

. utils/parse_options.sh

# 1. compile
if [ ! -d ${SPEECHX_BUILD} ]; then
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
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
    if [ ! -d $data/test ]; then
        # donwload dataset
        pushd $data
        wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_test.zip
        unzip  aishell_test.zip
        popd

        realpath $data/test/*/*.wav > $data/wavlist
        awk -F '/' '{ print $(NF) }' $data/wavlist | awk -F '.' '{ print $1 }' > $data/utt_id
        paste $data/utt_id $data/wavlist > $data/$aishell_wav_scp
    fi

    if [ ! -f $ckpt_dir/data/mean_std.json ]; then
        # download model
        mkdir -p $ckpt_dir
        pushd $ckpt_dir
        wget -c https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz
        tar xzfv asr0_deepspeech2_online_aishell_ckpt_0.2.0.model.tar.gz 
        popd
    fi

    lm=$data/zh_giga.no_cna_cmn.prune01244.klm
    if [ ! -f $lm ]; then
        # download kenlm bin
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


cmvn=$data/cmvn.ark
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # 3. convert cmvn format and compute linear feat
    cmvn_json2kaldi_main --json_file=$ckpt_dir/data/mean_std.json --cmvn_write_path=$cmvn

    ./local/split_data.sh $data $data/$aishell_wav_scp $aishell_wav_scp $nj

    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/feat.log \
    compute_linear_spectrogram_main \
        --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
        --feature_wspecifier=ark,scp:$data/split${nj}/JOB/feat.ark,$data/split${nj}/JOB/feat.scp \
        --cmvn_file=$cmvn \
    echo "feature make have finished!!!"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    #  decode w/o lm
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recog.wolm.log \
    ctc_beam_search_decoder_main \
        --feature_rspecifier=scp:$data/split${nj}/JOB/feat.scp \
        --model_path=$model_dir/avg_1.jit.pdmodel \
        --param_path=$model_dir/avg_1.jit.pdiparams \
        --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
	    --nnet_decoder_chunk=8 \
        --dict_file=$vocb_dir/vocab.txt \
        --result_wspecifier=ark,t:$data/split${nj}/JOB/result

    cat $data/split${nj}/*/result > $exp/${label_file}
    utils/compute-wer.py --char=1 --v=1 $text $exp/${label_file} > $exp/${wer}
    echo "ctc-prefix-beam-search-decoder-ol without lm has finished!!!"
    echo "please checkout in ${exp}/${wer}"
    tail -n 7 $exp/${wer}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # decode w/ ngram lm with feature input
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recog.lm.log \
    ctc_beam_search_decoder_main \
        --feature_rspecifier=scp:$data/split${nj}/JOB/feat.scp \
        --model_path=$model_dir/avg_1.jit.pdmodel \
        --param_path=$model_dir/avg_1.jit.pdiparams \
        --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
	    --nnet_decoder_chunk=8 \
        --dict_file=$vocb_dir/vocab.txt \
        --lm_path=$lm \
        --result_wspecifier=ark,t:$data/split${nj}/JOB/result_lm
 
    cat $data/split${nj}/*/result_lm > $exp/${label_file}_lm
    utils/compute-wer.py --char=1 --v=1 $text $exp/${label_file}_lm > $exp/${wer}.lm
    echo "ctc-prefix-beam-search-decoder-ol with lm test has finished!!!"
    echo "please checkout in ${exp}/${wer}.lm"
    tail -n 7 $exp/${wer}.lm
fi

wfst=$data/wfst/
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    mkdir -p $wfst
    if [ ! -f $wfst/aishell_graph.zip ]; then
        # download TLG graph
        pushd $wfst
        wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_graph.zip
        unzip aishell_graph.zip
        mv aishell_graph/* $wfst
        popd
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    #  decoder w/ TLG graph with feature input
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recog.wfst.log \
    ctc_tlg_decoder_main \
        --feature_rspecifier=scp:$data/split${nj}/JOB/feat.scp \
        --model_path=$model_dir/avg_1.jit.pdmodel \
        --param_path=$model_dir/avg_1.jit.pdiparams \
        --word_symbol_table=$wfst/words.txt \
        --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
        --graph_path=$wfst/TLG.fst --max_active=7500 \
	    --nnet_decoder_chunk=8 \
        --acoustic_scale=1.2 \
        --result_wspecifier=ark,t:$data/split${nj}/JOB/result_tlg

    cat $data/split${nj}/*/result_tlg > $exp/${label_file}_tlg
    utils/compute-wer.py --char=1 --v=1 $text $exp/${label_file}_tlg > $exp/${wer}.tlg
    echo "wfst-decoder-ol have finished!!!"
    echo "please checkout in ${exp}/${wer}.tlg"
    tail -n 7 $exp/${wer}.tlg
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    #  recognize from wav file w/ TLG graph
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recognizer.log \
    recognizer_main \
        --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
        --cmvn_file=$cmvn \
        --model_path=$model_dir/avg_1.jit.pdmodel \
        --param_path=$model_dir/avg_1.jit.pdiparams \
        --word_symbol_table=$wfst/words.txt \
	    --nnet_decoder_chunk=8 \
        --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
        --graph_path=$wfst/TLG.fst --max_active=7500 \
        --acoustic_scale=1.2 \
        --result_wspecifier=ark,t:$data/split${nj}/JOB/result_recognizer

    cat $data/split${nj}/*/result_recognizer > $exp/${label_file}_recognizer
    utils/compute-wer.py --char=1 --v=1 $text $exp/${label_file}_recognizer > $exp/${wer}.recognizer
    echo "recognizer test have finished!!!"
    echo "please checkout in ${exp}/${wer}.recognizer"
    tail -n 7 $exp/${wer}.recognizer
fi