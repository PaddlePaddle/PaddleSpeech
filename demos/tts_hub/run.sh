#!/bin/bash

if [ $# != 2 -a $# != 3 ];then
    echo "usage: CUDA_VISIBLE_DEVICES=0 ${0} text output_dir [lang]"
    exit -1
fi

ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
if [ ${ngpu} == 0 ];then
    device=cpu
else
    device=gpu
fi

echo "using ${device}..."

text=$1
output_dir=$2
if [ $# == 3 ];then
    lang=$3
else
    lang=zh
fi

if [ ! -d $output_dir ];then
    mkdir -p $output_dir
fi

python3 -u hub_infer.py \
--lang ${lang} \
--device ${device} \
--text \"${text}\" \
--output_dir ${output_dir}

exit 0
