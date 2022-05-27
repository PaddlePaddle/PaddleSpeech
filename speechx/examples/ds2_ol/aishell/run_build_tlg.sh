#!/bin/bash
set -eo pipefail

. path.sh

# attention, please replace the vocab is only for this script. 
# different acustic model has different vocab
ckpt_dir=data/fbank_model
unit=$ckpt_dir/data/lang_char/vocab.txt       # vocab file, line: char/spm_pice
model_dir=$ckpt_dir/exp/deepspeech2_online/checkpoints/

stage=-1
stop_stage=100
corpus=aishell
lexicon=data/lexicon.txt  # line: word ph0 ... phn, aishell/resource_aishell/lexicon.txt
text=data/text            # line: utt text, aishell/data_aishell/transcript/aishell_transcript_v0.8.txt

. utils/parse_options.sh

data=$PWD/data
mkdir -p $data

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
    if [ ! -f $data/speech.ngram.zh.tar.gz ];then
        pushd $data
        wget -c http://paddlespeech.bj.bcebos.com/speechx/examples/ngram/zh/speech.ngram.zh.tar.gz
        tar xvzf speech.ngram.zh.tar.gz
        popd
    fi

    if [ ! -f $ckpt_dir/data/mean_std.json ]; then
        mkdir -p $ckpt_dir
        pushd $ckpt_dir
        wget -c https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr0/WIP1_asr0_deepspeech2_online_wenetspeech_ckpt_1.0.0a.model.tar.gz
        tar xzfv WIP1_asr0_deepspeech2_online_wenetspeech_ckpt_1.0.0a.model.tar.gz
        popd
    fi
fi

if [ ! -f $unit ]; then
    echo "$0: No such file $unit"
    exit 1;
fi

if ! which ngram-count; then
    pushd $MAIN_ROOT/tools
    make srilm.done
    popd
fi

mkdir -p data/local/dict
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Prepare dict
    # line: char/spm_pices
    cp $unit data/local/dict/units.txt

    if [ ! -f $lexicon ];then
       utils/text_to_lexicon.py --has_key true --text $text --lexicon $lexicon
        echo "Generate $lexicon from $text"
    fi

    # filter by vocab
    # line: word ph0 ... phn -> line: word char0 ... charn
    utils/fst/prepare_dict.py \
        --unit_file $unit \
        --in_lexicon ${lexicon} \
        --out_lexicon data/local/dict/lexicon.txt
fi

lm=data/local/lm
mkdir -p $lm

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Train lm
    cp $text $lm/text
    local/aishell_train_lms.sh
    echo "build LM done."
fi

# build TLG
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # build T & L
  utils/fst/compile_lexicon_token_fst.sh \
      data/local/dict data/local/tmp data/local/lang
 
  # build G & TLG
  utils/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;

fi

aishell_wav_scp=aishell_test.scp
nj=40
cmvn=$data/cmvn_fbank.ark
wfst=$data/lang_test

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

    if [ ! -d $data/test ]; then
        pushd $data
        wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_test.zip
        unzip  aishell_test.zip
        popd

        realpath $data/test/*/*.wav > $data/wavlist
        awk -F '/' '{ print $(NF) }' $data/wavlist | awk -F '.' '{ print $1 }' > $data/utt_id
        paste $data/utt_id $data/wavlist > $data/$aishell_wav_scp
    fi

    ./local/split_data.sh $data $data/$aishell_wav_scp $aishell_wav_scp $nj

    cmvn-json2kaldi --json_file=$ckpt_dir/data/mean_std.json --cmvn_write_path=$cmvn
fi

wer=aishell_wer
label_file=aishell_result
export GLOG_logtostderr=1

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    #  TLG decoder
    utils/run.pl JOB=1:$nj $data/split${nj}/JOB/check_tlg.log \
    recognizer_main \
        --wav_rspecifier=scp:$data/split${nj}/JOB/${aishell_wav_scp} \
        --cmvn_file=$cmvn \
        --model_path=$model_dir/avg_5.jit.pdmodel \
        --streaming_chunk=30 \
        --use_fbank=true \
        --param_path=$model_dir/avg_5.jit.pdiparams \
        --word_symbol_table=$wfst/words.txt \
        --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
        --model_cache_shapes="5-1-2048,5-1-2048" \
        --graph_path=$wfst/TLG.fst --max_active=7500 \
        --acoustic_scale=1.2 \
        --result_wspecifier=ark,t:$data/split${nj}/JOB/result_check_tlg

    cat $data/split${nj}/*/result_check_tlg > $exp/${label_file}_check_tlg
    utils/compute-wer.py --char=1 --v=1 $text $exp/${label_file}_check_tlg > $exp/${wer}.check_tlg
    echo "recognizer test have finished!!!"
    echo "please checkout in ${exp}/${wer}.check_tlg"
fi

exit 0
