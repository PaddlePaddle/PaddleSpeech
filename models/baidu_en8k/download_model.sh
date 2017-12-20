#! /usr/bin/env bash

. ../../utils/utility.sh

URL='http://cloud.dlnel.org/filepub/?uuid=37a1c211-ec47-494c-973c-31437a10ae90'
MD5=5fe7639e720d51b3c3bdf7a1470c6272
TARGET=./baidu_en8k_model.tar.gz


echo "Download BaiduEn8k model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download BaiduEn8k model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
