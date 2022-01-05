#!/bin/bash
FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}
function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}
IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer']
MODE=$2

if [ ${MODE} = "lite_train_infer" ];then
    # pretrain lite train data
    curPath=$(readlink -f "$(dirname "$0")")
    cd ${curPath}/../../../examples/tiny/asr0
    source path.sh
    # download audio data
    bash ./local/data.sh || exit -1
    # download language model
    bash local/download_lm_en.sh
    if [ $? -ne 0 ]; then
    exit 1
    fi
    cd ${curPath}

elif [ ${MODE} = "whole_train_infer" ];then
    curPath=$(readlink -f "$(dirname "$0")")
    cd ${curPath}/../../../examples/aishell/asr0
    source path.sh
    # download audio data
    bash ./local/data.sh || exit -1
    # download language model
    bash local/download_lm_ch.sh
    if [ $? -ne 0 ]; then
    exit 1
    fi
    cd ${curPath}
elif [ ${MODE} = "whole_infer" ];then
    curPath=$(readlink -f "$(dirname "$0")")
    cd ${curPath}/../../../examples/aishell/asr0
    source path.sh
    # download audio data
    bash ./local/data.sh || exit -1
    # download language model
    bash local/download_lm_ch.sh
    if [ $? -ne 0 ]; then
    exit 1
    fi
    cd ${curPath}
else
    curPath=$(readlink -f "$(dirname "$0")")
    cd ${curPath}/../../../examples/aishell/asr0
    source path.sh
    # download audio data
    bash ./local/data.sh || exit -1
    # download language model
    bash local/download_lm_ch.sh
    if [ $? -ne 0 ]; then
    exit 1
    fi
    cd ${curPath}
fi
