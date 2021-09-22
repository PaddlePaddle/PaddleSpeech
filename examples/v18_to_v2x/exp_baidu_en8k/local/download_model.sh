#! /usr/bin/env bash
if [ $# != 1 ];then
    echo "usage: ${0} ckpt_dir"
    exit -1
fi

ckpt_dir=$1


. ${MAIN_ROOT}/utils/utility.sh

URL='https://deepspeech.bj.bcebos.com/eng_models/baidu_en8k_v1.8_to_v2.x.tar.gz'
MD5=fdabeb6c96963ac85d9188f0275c6a1b
TARGET=${ckpt_dir}/baidu_en8k_v1.8_to_v2.x.tar.gz


echo "Download BaiduEn8k model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download BaiduEn8k model!"
    exit 1
fi


exit 0
