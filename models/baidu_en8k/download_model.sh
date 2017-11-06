#! /usr/bin/env bash

. ../../utils/utility.sh

URL='To-be-added'
MD5=a19d40cb3b558eb696c44d883f32cfda
TARGET=./baidu_en8k_model.tar.gz


echo "Download BaiduEn8k model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download BaiduEn8k model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
