#!/bin/bash
set +x
set -e

export GLOG_logtostderr=1

. path.sh
#test websocket server 

model_dir=./resource/model
graph_dir=./resource/graph
cmvn=./data/cmvn.ark


#paddle_asr_online/resource.tar.gz
if [ ! -f $cmvn ]; then
    wget -c https://paddlespeech.bj.bcebos.com/s2t/paddle_asr_online/resource.tar.gz
    tar xzfv resource.tar.gz
    ln -s ./resource/data .
fi

websocket_server_main \
    --cmvn_file=$cmvn \
    --streaming_chunk=0.1 \
    --use_fbank=true \
    --model_path=$model_dir/avg_10.jit.pdmodel \
    --param_path=$model_dir/avg_10.jit.pdiparams \
    --model_cache_shapes="5-1-2048,5-1-2048" \
    --model_output_names=softmax_0.tmp_0,tmp_5,concat_0.tmp_0,concat_1.tmp_0 \
    --word_symbol_table=$graph_dir/words.txt \
    --graph_path=$graph_dir/TLG.fst --max_active=7500 \
    --port=8881 \
    --acoustic_scale=12 
