#!/bin/bash
set +x
set -e

. path.sh

nj=40
stage=0
stop_stage=5

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

ckpt_dir=$data/fbank_model
model_dir=$ckpt_dir/exp/deepspeech2_online/checkpoints/
vocb_dir=$ckpt_dir/data/lang_char/

# output
mkdir -p exp
exp=$PWD/exp

lm=$data/zh_giga.no_cna_cmn.prune01244.klm
aishell_wav_scp=aishell_test.scp
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
    if [ ! -d $data/test ]; then
        pushd $data
        wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_test.zip
        unzip  aishell_test.zip
        popd

        realpath $data/test/*/*.wav > $data/wavlist
        awk -F '/' '{ print $(NF) }' $data/wavlist | awk -F '.' '{ print $1 }' > $data/utt_id
        paste $data/utt_id $data/wavlist > $data/$aishell_wav_scp
    fi

    if [ ! -f $ckpt_dir/data/mean_std.json ]; then
        mkdir -p $ckpt_dir
        pushd $ckpt_dir
        wget -c https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr0/WIP1_asr0_deepspeech2_online_wenetspeech_ckpt_1.0.0a.model.tar.gz
        tar xzfv WIP1_asr0_deepspeech2_online_wenetspeech_ckpt_1.0.0a.model.tar.gz
        popd
    fi

    if [ ! -f $lm ]; then
        pushd $data
        wget -c https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
        popd
    fi
fi

# 3. make feature
text=$data/test/text
label_file=./aishell_result_fbank
wer=./aishell_wer_fbank

export GLOG_logtostderr=1


cmvn=$data/cmvn_fbank.ark
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # 3. convert cmvn format and compute fbank feat
    cmvn_json2kaldi_main --json_file=$ckpt_dir/data/mean_std.json --cmvn_write_path=$cmvn --binary=false

    ./local/split_data.sh $data $data/$aishell_wav_scp $aishell_wav_scp $nj

    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/feat.log \
    compute_fbank_main \
        --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
        --feature_wspecifier=ark,scp:$data/split${nj}/JOB/fbank_feat.ark,$data/split${nj}/JOB/fbank_feat.scp \
        --cmvn_file=$cmvn \
        --streaming_chunk=36
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    #  decode w/ lm by feature
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recog.fbank.wolm.log \
    ctc_beam_search_decoder_main \
        --feature_rspecifier=scp:$data/split${nj}/JOB/fbank_feat.scp \
        --model_path=$model_dir/avg_5.jit.pdmodel \
        --param_path=$model_dir/avg_5.jit.pdiparams \
        --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
    	--model_cache_shapes="5-1-2048,5-1-2048" \
	    --nnet_decoder_chunk=8 \
        --dict_file=$vocb_dir/vocab.txt \
        --result_wspecifier=ark,t:$data/split${nj}/JOB/result_fbank

    cat $data/split${nj}/*/result_fbank > $exp/${label_file}
    utils/compute-wer.py --char=1 --v=1 $text $exp/${label_file} > $exp/${wer}
    tail -n 7 $exp/${wer}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    #  decode with ngram lm by feature
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recog.fbank.lm.log \
    ctc_beam_search_decoder_main \
        --feature_rspecifier=scp:$data/split${nj}/JOB/fbank_feat.scp \
        --model_path=$model_dir/avg_5.jit.pdmodel \
        --param_path=$model_dir/avg_5.jit.pdiparams \
        --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
	    --model_cache_shapes="5-1-2048,5-1-2048" \
	    --nnet_decoder_chunk=8 \
        --dict_file=$vocb_dir/vocab.txt \
        --lm_path=$lm \
        --result_wspecifier=ark,t:$data/split${nj}/JOB/fbank_result_lm
 
    cat $data/split${nj}/*/fbank_result_lm > $exp/${label_file}_lm
    utils/compute-wer.py --char=1 --v=1 $text $exp/${label_file}_lm > $exp/${wer}.lm
    tail -n 7 $exp/${wer}.lm
fi

wfst=$data/wfst_fbank/
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    mkdir -p $wfst
    if [ ! -f $wfst/aishell_graph2.zip ]; then
        pushd $wfst
        wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_graph2.zip
        unzip aishell_graph2.zip
        mv aishell_graph2/* $wfst
        popd
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    #  decode w/ TLG graph by feature
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/recog.fbank.wfst.log \
    ctc_tlg_decoder_main \
        --feature_rspecifier=scp:$data/split${nj}/JOB/fbank_feat.scp \
        --model_path=$model_dir/avg_5.jit.pdmodel \
        --param_path=$model_dir/avg_5.jit.pdiparams \
        --word_symbol_table=$wfst/words.txt \
        --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
	    --model_cache_shapes="5-1-2048,5-1-2048" \
	    --nnet_decoder_chunk=8 \
        --graph_path=$wfst/TLG.fst --max_active=7500 \
        --acoustic_scale=1.2 \
        --result_wspecifier=ark,t:$data/split${nj}/JOB/result_tlg

    cat $data/split${nj}/*/result_tlg > $exp/${label_file}_tlg
    utils/compute-wer.py --char=1 --v=1 $text $exp/${label_file}_tlg > $exp/${wer}.tlg
    echo "wfst-decoder-ol have finished!!!"
    echo "please checkout in ${exp}/${wer}.tlg"
    tail -n 7 $exp/${wer}.tlg
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # recgonize w/ TLG graph by wav
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/fbank_recognizer.log \
    recognizer_main \
        --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
        --cmvn_file=$cmvn \
        --model_path=$model_dir/avg_5.jit.pdmodel \
        --use_fbank=true \
        --param_path=$model_dir/avg_5.jit.pdiparams \
        --word_symbol_table=$wfst/words.txt \
        --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
        --model_cache_shapes="5-1-2048,5-1-2048" \
	    --nnet_decoder_chunk=8 \
        --graph_path=$wfst/TLG.fst --max_active=7500 \
        --acoustic_scale=1.2 \
        --result_wspecifier=ark,t:$data/split${nj}/JOB/result_fbank_recognizer

    cat $data/split${nj}/*/result_fbank_recognizer > $exp/${label_file}_recognizer
    utils/compute-wer.py --char=1 --v=1 $text $exp/${label_file}_recognizer > $exp/${wer}.recognizer
    echo "recognizer test have finished!!!"
    echo "please checkout in ${exp}/${wer}.recognizer"
    tail -n 7 $exp/${wer}.recognizer
fi
