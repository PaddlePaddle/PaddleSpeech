#! /usr/bin/env bash

. ../../utils/utility.sh

URL='https://deepspeech.bj.bcebos.com/eng_models/baidu_en8k_v1.8_to_v2.x.tar.gz'
MD5=fdabeb6c96963ac85d9188f0275c6a1b
TARGET=./baidu_en8k_v1.8_to_v2.x.tar.gz


echo "Download BaiduEn8k model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download BaiduEn8k model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
