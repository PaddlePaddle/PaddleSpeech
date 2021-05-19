#!/bin/bash

if [ $# != 4 ];then
    echo "usage: $0 ckpt_prefix model_config mean_std vocab"
    exit -1
fi

ckpt_prefix=$1
model_config=$2
mean_std=$3
vocab=$4

output=release

mkdir -p ${output}
function clean() {
    rm -rf ${output}
}
trap clean EXIT

cp ${ckpt_prefix}.* ${output}
cp ${model_config} ${mean_std} ${vocab} ${output}

tar zcvf release.tar.gz ${output}

echo "tarball done!"
