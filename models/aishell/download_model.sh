#! /usr/bin/env bash

. ../../utils/utility.sh

URL='https://deepspeech.bj.bcebos.com/mandarin_models/aishell_model.tar.gz'
MD5=0ee83aa15fba421e5de8fc66c8feb350
TARGET=./aishell_model.tar.gz


echo "Download Aishell model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download Aishell model!"
    exit 1
fi
tar -zxvf $TARGET


exit 0
