#! /usr/bin/env bash

. ../../utils/utility.sh

URL='https://deepspeech.bj.bcebos.com/demo_models/baidu_en8k_model_fluid.tar.gz'
MD5=7e58fbf64aa4ecf639b049792ddcf788
TARGET=./baidu_en8k_model_fluid.tar.gz


echo "Download BaiduEn8k model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download BaiduEn8k model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
