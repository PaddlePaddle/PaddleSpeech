#!/bin/bash

if [ $# != 5 ];then
    echo "usage: $0 ckpt_prefix model_config mean_std vocab pack_name"
    exit -1
fi

ckpt_prefix=$1
model_config=$2
mean_std=$3
vocab=$4
pack_name=$5

output=release

mkdir -p ${output}
function clean() {
    rm -rf ${output}
}
trap clean EXIT

# ckpt_prfix dir
if [ -d ${ckpt_prefix} ];then
    cp -r ${ckpt_prefix} ${output}
fi
# ckpt_prfix.{json,...}
cp ${ckpt_prefix}.*  ${output}
# model config, mean std, vocab
cp ${model_config} ${mean_std} ${vocab} ${output}

tar zcvf ${pack_name}.release.tar.gz ${output}

echo "tarball: ${pack_name}.release.tar.gz done!"
