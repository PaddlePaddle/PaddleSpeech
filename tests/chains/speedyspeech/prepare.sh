#!/bin/bash
# usage bash prepare.sh MODE
# FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
MODE=$1

# dataline=$(cat ${FILENAME})

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
if [ ${MODE} = "lite_train_infer" ];then
    # pretrain lite train data
    wget -nc -P  ./pretrain_models/ https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip
    (cd ./pretrain_models && unzip pwg_baker_ckpt_0.4.zip)
    # download data
    rm -rf ./train_data/mini_BZNSYP
    wget -nc -P ./train_data/ https://paddlespeech.bj.bcebos.com/datasets/CE/speedyspeech_v0.5/mini_BZNSYP.tar.gz
    cd ./train_data/ && tar xzf mini_BZNSYP.tar.gz
    cd ../
elif [ ${MODE} = "whole_train_infer" ];then
    wget -nc -P  ./pretrain_models/ https://paddlespeech.bj.bcebos.com/Parakeet/released_models/speedyspeech/speedyspeech_nosil_baker_ckpt_0.5.zip
    wget -nc -P  ./pretrain_models/ https://paddlespeech.bj.bcebos.com/Parakeet/pwg_baker_ckpt_0.4.zip
    (cd ./pretrain_models && unzip speedyspeech_nosil_baker_ckpt_0.5.zip && unzip pwg_baker_ckpt_0.4.zip)
    rm -rf ./train_data/processed_BZNSYP
    wget -nc -P ./train_data/ https://paddlespeech.bj.bcebos.com/datasets/CE/speedyspeech_v0.5/processed_BZNSYP.tar.gz
    cd ./train_data/ && tar xzf processed_BZNSYP.tar.gz
    cd ../
else
    # whole infer using paddle inference library
    wget -nc -P  ./pretrain_models/ https://paddlespeech.bj.bcebos.com/Parakeet/speedyspeech_pwg_inference_0.5.zip
    (cd ./pretrain_models && unzip speedyspeech_pwg_inference_0.5.zip)
fi
