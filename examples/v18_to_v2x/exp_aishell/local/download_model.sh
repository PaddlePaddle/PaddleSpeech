#! /usr/bin/env bash

if [ $# != 1 ];then
    echo "usage: ${0} ckpt_dir"
    exit -1
fi

ckpt_dir=$1

. ${MAIN_ROOT}/utils/utility.sh

URL='https://deepspeech.bj.bcebos.com/mandarin_models/aishell_model_v1.8_to_v2.x.tar.gz'
MD5=4ade113c69ea291b8ce5ec6a03296659
TARGET=${ckpt_dir}/aishell_model_v1.8_to_v2.x.tar.gz


echo "Download Aishell model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download Aishell model!"
    exit 1
fi


exit 0
