#!/bin/bash
set +x
set -e

. path.sh


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


export GLOG_logtostderr=1

# 3. gen cmvn 
cmvn=$PWD/cmvn.ark
cmvn-json2kaldi --json_file=$ckpt_dir/data/mean_std.json --cmvn_write_path=$cmvn

text=$data/test/text
graph_dir=./aishell_graph
if [ ! -d $graph_dir ]; then
    wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/aishell_graph.zip
    unzip aishell_graph.zip 
fi

# 5. test websocket server 
websocket_server_main \
    --cmvn_file=$cmvn \
    --model_path=$model_dir/avg_1.jit.pdmodel \
    --streaming_chunk=0.1 \
    --convert2PCM32=true \
    --params_path=$model_dir/avg_1.jit.pdiparams \
    --word_symbol_table=$graph_dir/words.txt \
    --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
    --graph_path=$graph_dir/TLG.fst --max_active=7500 \
    --acoustic_scale=1.2 
